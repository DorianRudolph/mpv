// This filter was ported from Chromium (https://chromium.googlesource.com/chromium/chromium/+/51ed77e3f37a9a9b80d6d0a8259e84a8ca635259/media/filters/audio_renderer_algorithm.cc)
//
// Copyright 2015 The Chromium Authors. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//    * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//    * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <math.h>

#include "audio/aframe.h"
#include "audio/format.h"
#include "common/common.h"
#include "filters/f_autoconvert.h"
#include "filters/filter_internal.h"
#include "filters/user_filters.h"
#include "options/m_option.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct interval {
    int lo;
    int hi;
};

static bool in_interval(int n, struct interval q)
{
    return n >= q.lo && n <= q.hi;
}

static float **realloc_2d(float **p, int x, int y)
{
    float **array = realloc(p, sizeof(float*) * x + sizeof(float) * x * y);
    float* data = (float*) (array + x);
    for (int i = 0; i < x; ++i) {
        array[i] = data + i * y;
    }
    return array;
}

static void zero_2d(float **a, int x, int y)
{
    memset(a + x, 0, sizeof(float) * x * y);
}

static void zero_2d_partial(float **a, int x, int y)
{
    for (int i = 0; i < x; ++i) {
        memset(a[i], 0, sizeof(float) * y);
    }
}

// Energies of sliding windows of channels are interleaved.
// The number windows is |input_frames| - (|frames_per_window| - 1), hence,
// the method assumes |energy| must be, at least, of size
// (|input_frames| - (|frames_per_window| - 1)) * |channels|.
static void multi_channel_moving_block_energies(
    float **input, int input_frames, int channels,
    int frames_per_block, float *energy)
{
    int num_blocks = input_frames - (frames_per_block - 1);

    for (int k = 0; k < channels; ++k) {
        const float* input_channel = input[k];

        energy[k] = 0;

        // First block of channel |k|.
        for (int m = 0; m < frames_per_block; ++m) {
            energy[k] += input_channel[m] * input_channel[m];
        }

        const float* slide_out = input_channel;
        const float* slide_in = input_channel + frames_per_block;
        for (int n = 1; n < num_blocks; ++n, ++slide_in, ++slide_out) {
            energy[k + n * channels] = energy[k + (n - 1) * channels]
                - *slide_out * *slide_out + *slide_in * *slide_in;
        }
    }
}

static float multi_channel_similarity_measure(
    const float* dot_prod_a_b,
    const float* energy_a, const float* energy_b,
    int channels)
{
    const float epsilon = 1e-12f;
    float similarity_measure = 0.0f;
    for (int n = 0; n < channels; ++n) {
        similarity_measure += dot_prod_a_b[n]
            / sqrtf(energy_a[n] * energy_b[n] + epsilon);
    }
    return similarity_measure;
}

// Dot-product of channels of two AudioBus. For each AudioBus an offset is
// given. |dot_product[k]| is the dot-product of channel |k|. The caller should
// allocate sufficient space for |dot_product|.
static void multi_channel_dot_product(
    float **a, int frame_offset_a,
    float **b, int frame_offset_b,
    int channels,
    int num_frames, float *dot_product)
{
    assert(frame_offset_a >= 0);
    assert(frame_offset_b >= 0);

    memset(dot_product, 0, sizeof(*dot_product) * channels);
    for (int k = 0; k < channels; ++k) {
        const float* ch_a = a[k] + frame_offset_a;
        const float* ch_b = b[k] + frame_offset_b;
        for (int n = 0; n < num_frames; ++n) {
            dot_product[k] += *ch_a++ * *ch_b++;
        }
    }
}

// Fit the curve f(x) = a * x^2 + b * x + c such that
//   f(-1) = y[0]
//   f(0) = y[1]
//   f(1) = y[2]
// and return the maximum, assuming that y[0] <= y[1] >= y[2].
static void quadratic_interpolation(
    const float* y_values, float* extremum, float* extremum_value)
{
    float a = 0.5f * (y_values[2] + y_values[0]) - y_values[1];
    float b = 0.5f * (y_values[2] - y_values[0]);
    float c = y_values[1];

    if (a == 0.f) {
        // The coordinates are colinear (within floating-point error).
        *extremum = 0;
        *extremum_value = y_values[1];
    } else {
        *extremum = -b / (2.f * a);
        *extremum_value = a * (*extremum) * (*extremum) + b * (*extremum) + c;
    }
}

// Search a subset of all candid blocks. The search is performed every
// |decimation| frames. This reduces complexity by a factor of about
// 1 / |decimation|. A cubic interpolation is used to have a better estimate of
// the best match.
static int decimated_search(
    int decimation, struct interval exclude_interval,
    float **target_block, int target_block_frames,
    float **search_segment, int search_segment_frames,
    int channels,
    const float *energy_target_block, const float *energy_candidate_blocks)
{
    int num_candidate_blocks = search_segment_frames - (target_block_frames - 1);
    float dot_prod [MP_NUM_CHANNELS];
    float similarity[3];  // Three elements for cubic interpolation.

    int n = 0;
    multi_channel_dot_product(
        target_block, 0,
        search_segment, n,
        channels,
        target_block_frames, dot_prod);
    similarity[0] = multi_channel_similarity_measure(
        dot_prod, energy_target_block,
        &energy_candidate_blocks[n * channels], channels);

    // Set the starting point as optimal point.
    float best_similarity = similarity[0];
    int optimal_index = 0;

    n += decimation;
    if (n >= num_candidate_blocks) {
        return 0;
    }

    multi_channel_dot_product(
        target_block, 0,
        search_segment, n,
        channels,
        target_block_frames, dot_prod);
    similarity[1] = multi_channel_similarity_measure(
        dot_prod, energy_target_block,
        &energy_candidate_blocks[n * channels], channels);

    n += decimation;
    if (n >= num_candidate_blocks) {
        // We cannot do any more sampling. Compare these two values and return the
        // optimal index.
        return similarity[1] > similarity[0] ? decimation : 0;
    }

    for (; n < num_candidate_blocks; n += decimation) {
        multi_channel_dot_product(
            target_block, 0,
            search_segment, n,
            channels,
            target_block_frames, dot_prod);

        similarity[2] = multi_channel_similarity_measure(
            dot_prod, energy_target_block,
            &energy_candidate_blocks[n * channels], channels);

        if ((similarity[1] > similarity[0] && similarity[1] >= similarity[2]) ||
            (similarity[1] >= similarity[0] && similarity[1] > similarity[2])) {
            // A local maximum is found. Do a cubic interpolation for a better
            // estimate of candidate maximum.
            float normalized_candidate_index;
            float candidate_similarity;
            quadratic_interpolation(similarity, &normalized_candidate_index,
                                    &candidate_similarity);

            int candidate_index = n - decimation
                 + (int)(normalized_candidate_index * decimation +  0.5f);
            if (candidate_similarity > best_similarity
                && !in_interval(candidate_index, exclude_interval)) {
                optimal_index = candidate_index;
                best_similarity = candidate_similarity;
            }
        } else if (n + decimation >= num_candidate_blocks &&
                   similarity[2] > best_similarity &&
                   !in_interval(n, exclude_interval)) {
            // If this is the end-point and has a better similarity-measure than
            // optimal, then we accept it as optimal point.
            optimal_index = n;
            best_similarity = similarity[2];
        }
        memmove(similarity, &similarity[1], 2 * sizeof(*similarity));
    }
    return optimal_index;
}

// Search [|low_limit|, |high_limit|] of |search_segment| to find a block that
// is most similar to |target_block|. |energy_target_block| is the energy of the
// |target_block|. |energy_candidate_blocks| is the energy of all blocks within
// |search_block|.
static int full_search(
        int low_limit, int high_limit,
        struct interval exclude_interval,
        float **target_block, int target_block_frames,
        float **search_block, int search_block_frames,
        int channels,
        const float* energy_target_block,
        const float* energy_candidate_blocks)
{
    // int block_size = target_block->frames;
    float dot_prod [sizeof(float) * MP_NUM_CHANNELS];

    float best_similarity = -FLT_MAX;//FLT_MIN;
    int optimal_index = 0;

    for (int n = low_limit; n <= high_limit; ++n) {
        if (in_interval(n, exclude_interval)) {
            continue;
        }
        multi_channel_dot_product(target_block, 0, search_block, n, channels,
            target_block_frames, dot_prod);

        float similarity = multi_channel_similarity_measure(
            dot_prod, energy_target_block,
            &energy_candidate_blocks[n * channels], channels);

        if (similarity > best_similarity) {
            best_similarity = similarity;
            optimal_index = n;
        }
    }

    return optimal_index;
}

// Find the index of the block, within |search_block|, that is most similar
// to |target_block|. Obviously, the returned index is w.r.t. |search_block|.
// |exclude_interval| is an interval that is excluded from the search.
static int compute_optimal_index(
        float **search_block, int search_block_frames,
        float **target_block, int target_block_frames,
        float *energy_candidate_blocks,
        int channels,
        struct interval exclude_interval)
{
    int num_candidate_blocks = search_block_frames - (target_block_frames - 1);

    // This is a compromise between complexity reduction and search accuracy. I
    // don't have a proof that down sample of order 5 is optimal.
    // One can compute a decimation factor that minimizes complexity given
    // the size of |search_block| and |target_block|. However, my experiments
    // show the rate of missing the optimal index is significant.
    // This value is chosen heuristically based on experiments.
    const int search_decimation = 5;

    float energy_target_block [MP_NUM_CHANNELS];
    // energy_candidate_blocks must have at least size
    // sizeof(float) * channels * num_candidate_blocks

    // Energy of all candid frames.
    multi_channel_moving_block_energies(
        search_block,
        search_block_frames,
        channels,
        target_block_frames,
        energy_candidate_blocks);

    // Energy of target frame.
    multi_channel_dot_product(
        target_block, 0,
        target_block, 0,
        channels,
        target_block_frames, energy_target_block);

    int optimal_index = decimated_search(
        search_decimation, exclude_interval,
        target_block, target_block_frames,
        search_block, search_block_frames,
        channels,
        energy_target_block,
        energy_candidate_blocks);

    int lim_low = MPMAX(0, optimal_index - search_decimation);
    int lim_high = MPMIN(num_candidate_blocks - 1,
                            optimal_index + search_decimation);
    return full_search(
        lim_low, lim_high, exclude_interval,
        target_block, target_block_frames,
        search_block, search_block_frames,
        channels,
        energy_target_block, energy_candidate_blocks);
}

// Waveform Similarity Overlap-and-add (WSOLA).
//
// One WSOLA iteration
//
// 1) Extract |target_block| as input frames at indices
//    [|target_block_index|, |target_block_index| + |ola_window_size|).
//    Note that |target_block| is the "natural" continuation of the output.
//
// 2) Extract |search_block| as input frames at indices
//    [|search_block_index|,
//     |search_block_index| + |num_candidate_blocks| + |ola_window_size|).
//
// 3) Find a block within the |search_block| that is most similar
//    to |target_block|. Let |optimal_index| be the index of such block and
//    write it to |optimal_block|.
//
// 4) Update:
//    |optimal_block| = |transition_window| * |target_block| +
//    (1 - |transition_window|) * |optimal_block|.
//
// 5) Overlap-and-add |optimal_block| to the |wsola_output|.
//
// 6) Update:write

struct f_opts {
    // Max/min supported playback rates for fast/slow audio. Audio outside of these
    // ranges are muted.
    // Audio at these speeds would sound better under a frequency domain algorithm.
    float min_playback_rate;
    float max_playback_rate;
    // Overlap-and-add window size in milliseconds.
    float ola_window_size_ms;
    // Size of search interval in milliseconds. The search interval is
    // [-delta delta] around |output_index| * |playback_rate|. So the search
    // interval is 2 * delta.
    float wsola_search_interval_ms;
};

struct priv {
    struct f_opts *opts;

    struct mp_pin *in_pin;
    struct mp_aframe *cur_format;
    struct mp_aframe_pool *out_pool;
    bool sent_final;
    struct mp_aframe *pending;
    bool initialized;
    float speed;
    double frame_delay;

    // Number of channels in audio stream.
    int channels;
    // Sample rate of audio stream.
    int samples_per_second;
    // If muted, keep track of partial frames that should have been skipped over.
    double muted_partial_frame;
    // Book keeping of the current time of generated audio, in frames. This
    // should be appropriately updated when out samples are generated, regardless
    // of whether we push samples out when fill_buffer() is called or we store
    // audio in |wsola_output| for the subsequent calls to fill_buffer().
    // Furthermore, if samples from |audio_buffer| are evicted then this
    // member variable should be updated based on |playback_rate|.
    // Note that this member should be updated ONLY by calling update_output_time(),
    // so that |search_block_index| is update accordingly.
    double output_time;
    // The offset of the center frame of |search_block| w.r.t. its first frame.
    int search_block_center_offset;
    // Index of the beginning of the |search_block|, in frames.
    int search_block_index;
    // Number of Blocks to search to find the most similar one to the target
    // frame.
    int num_candidate_blocks;
    // Index of the beginning of the target block, counted in frames.
    int target_block_index;
    // Overlap-and-add window size in frames.
    int ola_window_size;
    // The hop size of overlap-and-add in frames. This implementation assumes 50%
    // overlap-and-add.
    int ola_hop_size;
    // Number of frames in |wsola_output| that overlap-and-add is completed for
    // them and can be copied to output if fill_buffer() is called. It also
    // specifies the index where the next WSOLA window has to overlap-and-add.
    int num_complete_frames;
    // Overlap-and-add window.
    float *ola_window;
    // Transition window, used to update |optimal_block| by a weighted sum of
    // |optimal_block| and |target_block|.
    float *transition_window;
    // This stores a part of the output that is created but couldn't be rendered.
    // Output is generated frame-by-frame which at some point might exceed the
    // number of requested samples. Furthermore, due to overlap-and-add,
    // the last half-window of the output is incomplete, which is stored in this
    // buffer.
    float **wsola_output;
    int wsola_output_size;
    // Auxiliary variables to avoid allocation in every iteration.
    // Stores the optimal block in every iteration. This is the most
    // similar block to |target_block| within |search_block| and it is
    // overlap-and-added to |wsola_output|.
    float **optimal_block;
    // A block of data that search is performed over to find the |optimal_block|.
    float **search_block;
    int search_block_size;
    // Stores the target block, denoted as |target| above. |search_block| is
    // searched for a block (|optimal_block|) that is most similar to
    // |target_block|.
    float **target_block;
    // Buffered audio data.
    float **input_buffer;
    int input_buffer_size;
    int input_buffer_frames;
    float *energy_candidate_blocks;
};

static void peek_buffer(struct priv *p,
    int frames, int read_offset, int write_offset, float **dest)
{
    assert(p->input_buffer_frames >= frames);
    for (int i = 0; i < p->channels; ++i) {
        memcpy(dest[i] + write_offset,
            p->input_buffer[i] + read_offset,
            frames * sizeof(float));
    }
}

static void seek_buffer(struct priv *p, int frames)
{
    assert(p->input_buffer_frames >= frames);
    p->input_buffer_frames -= frames;
    for (int i = 0; i < p->channels; ++i) {
        memmove(p->input_buffer[i], p->input_buffer[i] + frames,
            p->input_buffer_frames * sizeof(float));
    }
}

static void read_buffer(struct priv *p, int frames, float **dest)
{
    peek_buffer(p, frames, 0, 0, dest);
    seek_buffer(p, frames);
}

static int write_completed_frames_to(
    struct priv *p, int requested_frames, int dest_offset, float **dest)
{
    int rendered_frames = MPMIN(p->num_complete_frames, requested_frames);

    if (rendered_frames == 0)
        return 0;  // There is nothing to read from |wsola_output|, return.

    for (int i = 0; i < p->channels; ++i) {
        memcpy(dest[i] + dest_offset, p->wsola_output[i],
            rendered_frames * sizeof(float));
    }

    // Remove the frames which are read.
    int frames_to_move = p->wsola_output_size - rendered_frames;
    for (int k = 0; k < p->channels; ++k) {
        float *ch = p->wsola_output[k];
        memmove(ch, &ch[rendered_frames], sizeof(*ch) * frames_to_move);
    }
    p->num_complete_frames -= rendered_frames;
    return rendered_frames;
}

static bool can_perform_wsola(struct priv *p)
{
    const int search_block_size = p->num_candidate_blocks
        + (p->ola_window_size - 1);
    return p->target_block_index + p->ola_window_size <= p->input_buffer_frames
        && p->search_block_index + search_block_size <= p->input_buffer_frames;
}

// number of frames needed until a wsola iteration can be performed
static int frames_needed(struct priv *p)
{
    return MPMAX(0, MPMAX(
        p->target_block_index + p->ola_window_size - p->input_buffer_frames,
        p->search_block_index + p->search_block_size - p->input_buffer_frames));
}

static int fill_input_buffer(struct priv *p, bool final)
{
    int needed = frames_needed(p);
    int frame_size = mp_aframe_get_size(p->pending);
    int read = MPMIN(needed, frame_size);
    int total_fill = final ? needed : read;
    if (total_fill == 0) return 0;

    assert(total_fill + p->input_buffer_frames <= p->input_buffer_size);

    uint8_t **planes = mp_aframe_get_data_ro(p->pending);
    for (int i = 0; i < p->channels; ++i) {
        memcpy(p->input_buffer[i] + p->input_buffer_frames,
            planes[i], read * sizeof(float));
        for (int j = read; j < total_fill; ++j) {
            p->input_buffer[p->input_buffer_frames + j] = 0;
        }
    }

    p->input_buffer_frames += total_fill;
    return read;
}

static bool target_is_within_search_region(struct priv *p)
{
    const int search_block_size = p->num_candidate_blocks + (p->ola_window_size - 1);

    return p->target_block_index >= p->search_block_index
        && p->target_block_index + p->ola_window_size
            <= p->search_block_index + search_block_size;
}


static void peek_audio_with_zero_prepend(
    struct priv *p, int read_offset_frames, float **dest, int dest_frames)
{
    assert(read_offset_frames + dest_frames <= p->input_buffer_frames);

    int write_offset = 0;
    int num_frames_to_read = dest_frames;
    if (read_offset_frames < 0) {
        int num_zero_frames_appended = MPMIN(
            -read_offset_frames, num_frames_to_read);
        read_offset_frames = 0;
        num_frames_to_read -= num_zero_frames_appended;
        write_offset = num_zero_frames_appended;
        zero_2d_partial(dest, p->channels, num_zero_frames_appended);
    }
    peek_buffer(p, num_frames_to_read, read_offset_frames, write_offset, dest);
}

static void get_optimal_block(struct priv *p)
{
    int optimal_index = 0;

    // An interval around last optimal block which is excluded from the search.
    // This is to reduce the buzzy sound. The number 160 is rather arbitrary and
    // derived heuristically.
    const int exclude_interval_length_frames = 160;
    if (target_is_within_search_region(p)) {
        optimal_index = p->target_block_index;
        peek_audio_with_zero_prepend(p,
            optimal_index, p->optimal_block, p->ola_window_size);
    } else {
        peek_audio_with_zero_prepend(p,
            p->target_block_index, p->target_block, p->ola_window_size);
        peek_audio_with_zero_prepend(p,
            p->search_block_index, p->search_block, p->search_block_size);
        int last_optimal = p->target_block_index
            - p->ola_hop_size - p->search_block_index;
        struct interval exclude_iterval = {
            .lo = last_optimal - exclude_interval_length_frames / 2,
            .hi = last_optimal + exclude_interval_length_frames / 2
        };

        // |optimal_index| is in frames and it is relative to the beginning of the
        // |search_block|.
        optimal_index = compute_optimal_index(
            p->search_block, p->search_block_size,
            p->target_block, p->ola_window_size,
            p->energy_candidate_blocks,
            p->channels,
            exclude_iterval);

        // Translate |index| w.r.t. the beginning of |audio_buffer| and extract the
        // optimal block.
        optimal_index += p->search_block_index;
        peek_audio_with_zero_prepend(p,
            optimal_index, p->optimal_block, p->ola_window_size);

        // Make a transition from target block to the optimal block if different.
        // Target block has the best continuation to the current output.
        // Optimal block is the most similar block to the target, however, it might
        // introduce some discontinuity when over-lap-added. Therefore, we combine
        // them for a smoother transition. The length of transition window is twice
        // as that of the optimal-block which makes it like a weighting function
        // where target-block has higher weight close to zero (weight of 1 at index
        // 0) and lower weight close the end.
        for (int k = 0; k < p->channels; ++k) {
            float* ch_opt = p->optimal_block[k];
            float* ch_target = p->target_block[k];
            for (int n = 0; n < p->ola_window_size; ++n) {
                ch_opt[n] = ch_opt[n] * p->transition_window[n]
                    + ch_target[n] * p->transition_window[p->ola_window_size + n];
            }
        }
    }

    // Next target is one hop ahead of the current optimal.
    p->target_block_index = optimal_index + p->ola_hop_size;
}

static void update_output_time(
    struct priv *p, float playback_rate, double time_change)
{
    p->output_time += time_change;
    // Center of the search region, in frames.
    int search_block_center_index = (int)(p->output_time * playback_rate + 0.5);
    p->search_block_index = search_block_center_index
        - p->search_block_center_offset;
}

static void remove_old_input_frames(struct priv *p, float playback_rate)
{
    const int earliest_used_index = MPMIN(
        p->target_block_index, p->search_block_index);
    if (earliest_used_index <= 0)
        return;  // Nothing to remove.

    // Remove frames from input and adjust indices accordingly.
    seek_buffer(p, earliest_used_index);
    p->target_block_index -= earliest_used_index;

    // Adjust output index.
    double output_time_change = ((double) earliest_used_index) / playback_rate;
    assert(p->output_time >= output_time_change);
    update_output_time(p, playback_rate, -output_time_change);
}

static bool run_one_wsola_iteration(struct priv *p, float playback_rate)
{
    if (!can_perform_wsola(p)){
        return false;
    }

    get_optimal_block(p);

    // Overlap-and-add.
    for (int k = 0; k < p->channels; ++k) {
        float* ch_opt_frame = p->optimal_block[k];
        float* ch_output = p->wsola_output[k] + p->num_complete_frames;
        for (int n = 0; n < p->ola_hop_size; ++n) {
            ch_output[n] = ch_output[n] * p->ola_window[p->ola_hop_size + n] +
                ch_opt_frame[n] * p->ola_window[n];
        }

        // Copy the second half to the output.
        memcpy(&ch_output[p->ola_hop_size], &ch_opt_frame[p->ola_hop_size],
               sizeof(*ch_opt_frame) * p->ola_hop_size);
    }

    p->num_complete_frames += p->ola_hop_size;
    update_output_time(p,
        playback_rate, p->ola_hop_size);
    remove_old_input_frames(p, playback_rate);
    return true;
}

static int fill_buffer(struct priv *p,
    float **dest, int dest_size, float playback_rate)
{
    if (playback_rate == 0) return 0;

    // Optimize the muted case to issue a single clear instead of performing
    // the full crossfade and clearing each crossfaded frame.
    if (playback_rate < p->opts->min_playback_rate
        || (playback_rate > p->opts->max_playback_rate
            && p->opts->max_playback_rate > 0)) {
        int frames_to_render = MPMIN(dest_size,
            (int) (p->input_buffer_frames / playback_rate));

        // Compute accurate number of frames to actually skip in the source data.
        // Includes the leftover partial frame from last request. However, we can
        // only skip over complete frames, so a partial frame may remain for next
        // time.
        p->muted_partial_frame += frames_to_render * playback_rate;
        int seek_frames = (int) (p->muted_partial_frame);
        zero_2d_partial(dest, p->channels, frames_to_render);
        seek_buffer(p, seek_frames);

        // Determine the partial frame that remains to be skipped for next call. If
        // the user switches back to playing, it may be off time by this partial
        // frame, which would be undetectable. If they subsequently switch to
        // another playback rate that mutes, the code will attempt to line up the
        // frames again.
        p->muted_partial_frame -= seek_frames;
        return frames_to_render;
    }

    int slower_step = (int) ceilf(p->ola_window_size * playback_rate);
    int faster_step = (int) ceilf(p->ola_window_size / playback_rate);

    // Optimize the most common |playback_rate| ~= 1 case to use a single copy
    // instead of copying frame by frame.
    if (p->ola_window_size <= faster_step && slower_step >= p->ola_window_size) {
        int frames_to_copy = MPMIN(dest_size, p->input_buffer_frames);
        read_buffer(p, frames_to_copy, dest);
        return frames_to_copy;
    }

    int rendered_frames = 0;
    do {
        rendered_frames += write_completed_frames_to(p,
            dest_size - rendered_frames, rendered_frames, dest);
    } while (rendered_frames < dest_size
             && run_one_wsola_iteration(p, playback_rate));
    return rendered_frames;
}

static bool frames_available(struct priv *p)
{
    return can_perform_wsola(p) || p->num_complete_frames > 0;
}

static bool init_scaletempo2(struct mp_filter *f);
static void reset(struct mp_filter *f);

static void process(struct mp_filter *f)
{
    struct priv *p = f->priv;

    if (!mp_pin_in_needs_data(f->ppins[1]))
        return;

    while (!p->initialized || !p->pending || !frames_available(p)) {

        bool eof = false;
        if (!p->pending || !mp_aframe_get_size(p->pending)) {
            struct mp_frame frame = mp_pin_out_read(p->in_pin);
            if (frame.type == MP_FRAME_AUDIO) {
                TA_FREEP(&p->pending);
                p->pending = frame.data;
            } else if (frame.type == MP_FRAME_EOF) {
                eof = true;
            } else if (frame.type) {
                MP_ERR(f, "unexpected frame type\n");
                goto error;
            } else {
                return; // no new data yet
            }
        }
        assert(p->pending || eof);

        if (!p->initialized) {
            if (!p->pending) {
                mp_pin_in_write(f->ppins[1], MP_EOF_FRAME);
                return;
            }
            if (!init_scaletempo2(f))
                goto error;
        }

        bool format_change =
            p->pending && !mp_aframe_config_equals(p->pending, p->cur_format);

        bool final = format_change || eof;
        if (p->pending && !format_change && !p->sent_final) {
            int read = fill_input_buffer(p, final);
            p->frame_delay += read;
            mp_aframe_skip_samples(p->pending, read);
        }
        p->sent_final |= final;

        if (frames_available(p)) {
            if (eof) {
                mp_pin_out_repeat_eof(p->in_pin); // drain more next time
            }
        } else if (final) {
            p->initialized = false;
            p->sent_final = false;
            if (eof) {
                mp_pin_in_write(f->ppins[1], MP_EOF_FRAME);
                return;
            } else if (format_change) {
                // go on with proper reinit on the next iteration
                p->initialized = false;
                p->sent_final = false;
            }
        }
    }

    assert(p->pending);
    if (frames_available(p)) {
        struct mp_aframe *out = mp_aframe_new_ref(p->cur_format);
        int out_samples = p->ola_hop_size;
        if (mp_aframe_pool_allocate(p->out_pool, out, out_samples) < 0) {
            talloc_free(out);
            goto error;
        }

        mp_aframe_copy_attributes(out, p->pending);

        uint8_t **planes = mp_aframe_get_data_rw(out);
        assert(planes);
        assert(mp_aframe_get_planes(out) == p->channels);

        out_samples = fill_buffer(p, (float**)planes, out_samples, p->speed);

        double pts = mp_aframe_get_pts(p->pending);
        p->frame_delay -= out_samples * p->speed;

        if (pts != MP_NOPTS_VALUE) {
            double delay = p->frame_delay / mp_aframe_get_effective_rate(out);
            mp_aframe_set_pts(out, pts - delay);
        }

        mp_aframe_set_size(out, out_samples);
        mp_aframe_mul_speed(out, p->speed);
        mp_pin_in_write(f->ppins[1], MAKE_FRAME(MP_FRAME_AUDIO, out));
    }

    return;
error:
    mp_filter_internal_mark_failed(f);
}

// Return a "periodic" Hann window. This is the first L samples of an L+1
// Hann window. It is perfect reconstruction for overlap-and-add.
static void get_symmetric_hanning_window(int window_length, float* window)
{
  const float scale = 2.0f * M_PI / window_length;
  for (int n = 0; n < window_length; ++n)
    window[n] = 0.5f * (1.0f - cosf(n * scale));
}

static bool init_scaletempo2(struct mp_filter *f)
{
    struct priv *p = f->priv;

    assert(p->pending);

    if (mp_aframe_get_format(p->pending) != AF_FORMAT_FLOATP)
        return false;

    mp_aframe_reset(p->cur_format);

    p->muted_partial_frame = 0;
    p->output_time = 0;
    p->search_block_center_offset = 0;
    p->search_block_index = 0;
    p->num_complete_frames = 0;
    p->channels = mp_aframe_get_channels(p->pending);

    p->samples_per_second = mp_aframe_get_rate(p->pending);
    p->num_candidate_blocks = (int)(p->opts->wsola_search_interval_ms 
        * p->samples_per_second / 1000);
    p->ola_window_size = (int)(p->opts->ola_window_size_ms 
        * p->samples_per_second / 1000);
    // Make sure window size in an even number.
    p->ola_window_size += p->ola_window_size & 1;
    p->ola_hop_size = p->ola_window_size / 2;
    // |num_candidate_blocks| / 2 is the offset of the center of the search
    // block to the center of the first (left most) candidate block. The offset
    // of the center of a candidate block to its left most point is
    // |ola_window_size| / 2 - 1. Note that |ola_window_size| is even and in
    // our convention the center belongs to the left half, so we need to subtract
    // one frame to get the correct offset.
    //
    //                             Search Block
    //              <------------------------------------------->
    //
    //   |ola_window_size| / 2 - 1
    //              <----
    //
    //             |num_candidate_blocks| / 2
    //                   <----------------
    //                                 center
    //              X----X----------------X---------------X-----X
    //              <---------->                     <---------->
    //                Candidate      ...               Candidate
    //                   1,          ...         |num_candidate_blocks|
    p->search_block_center_offset = p->num_candidate_blocks / 2
        + (p->ola_window_size / 2 - 1);
    p->ola_window = realloc(p->ola_window, sizeof(float) * p->ola_window_size);
    get_symmetric_hanning_window(p->ola_window_size, p->ola_window);
    p->transition_window = realloc(p->transition_window,
        sizeof(float) * p->ola_window_size * 2);
    get_symmetric_hanning_window(2 * p->ola_window_size, p->transition_window);

    p->wsola_output_size = p->ola_window_size + p->ola_hop_size;
    p->wsola_output = realloc_2d(p->wsola_output, p->channels, p->wsola_output_size);
    // Initialize for overlap-and-add of the first block.
    zero_2d(p->wsola_output, p->channels, p->wsola_output_size);

    // Auxiliary containers.
    p->optimal_block = realloc_2d(p->optimal_block, p->channels, p->ola_window_size);
    p->search_block_size = p->num_candidate_blocks + (p->ola_window_size - 1);
    p->search_block = realloc_2d(p->search_block, p->channels, p->search_block_size);
    p->target_block = realloc_2d(p->target_block, p->channels, p->ola_window_size);

    p->initialized = true;
    p->sent_final = false;
    p->input_buffer_size = 4 * MPMAX(p->ola_window_size, p->search_block_size);
    p->input_buffer = realloc_2d(p->input_buffer, p->channels, p->input_buffer_size);
    p->input_buffer_frames = 0;
    p->frame_delay = 0;

    p->energy_candidate_blocks = realloc(p->energy_candidate_blocks,
        sizeof(float) * p->channels * p->num_candidate_blocks);

    mp_aframe_config_copy(p->cur_format, p->pending);

    return true;
}

static bool command(struct mp_filter *f, struct mp_filter_command *cmd)
{
    struct priv *p = f->priv;

    switch (cmd->type) {
    case MP_FILTER_COMMAND_SET_SPEED:
        p->speed = cmd->speed;
        return true;
    }

    return false;
}

static void reset(struct mp_filter *f)
{
    struct priv *p = f->priv;

    // Clear the queue of decoded packets (releasing the buffers).
    p->input_buffer_frames = 0;
    p->output_time = 0.0;
    p->search_block_index = 0;
    p->target_block_index = 0;
    zero_2d(p->wsola_output, p->channels, p->wsola_output_size);
    p->num_complete_frames = 0;
    p->frame_delay = 0;

    p->initialized = false;

    TA_FREEP(&p->pending);
}

static void destroy(struct mp_filter *f)
{
    struct priv *p = f->priv;
    free(p->ola_window);
    free(p->transition_window);
    free(p->wsola_output);
    free(p->optimal_block);
    free(p->search_block);
    free(p->target_block);
    free(p->input_buffer);
    free(p->energy_candidate_blocks);
    talloc_free(p->pending);
}

static const struct mp_filter_info af_scaletempo2_filter = {
    .name = "scaletempo2",
    .priv_size = sizeof(struct priv),
    .process = process,
    .command = command,
    .reset = reset,
    .destroy = destroy,
};

static struct mp_filter *af_scaletempo2_create(
    struct mp_filter *parent, void *options)
{
    struct mp_filter *f = mp_filter_create(parent, &af_scaletempo2_filter);
    if (!f) {
        talloc_free(options);
        return NULL;
    }

    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");

    struct priv *p = f->priv;
    p->opts = talloc_steal(p, options);
    p->speed = 1.0;
    p->frame_delay = 0;
    p->cur_format = talloc_steal(p, mp_aframe_create());
    p->out_pool = mp_aframe_pool_create(p);
    p->ola_window = NULL;
    p->transition_window = NULL;

    p->pending = NULL;
    p->initialized = false;

    struct mp_autoconvert *conv = mp_autoconvert_create(f);
    if (!conv)
        abort();

    mp_autoconvert_add_afmt(conv, AF_FORMAT_FLOATP);

    mp_pin_connect(conv->f->pins[0], f->ppins[0]);
    p->in_pin = conv->f->pins[1];

    return f;
}

#define OPT_BASE_STRUCT struct f_opts
const struct mp_user_filter_entry af_scaletempo2 = {
    .desc = {
        .description = "Scale audio tempo while maintaining pitch"
            " (filter ported from chromium)",
        .name = "scaletempo2",
        .priv_size = sizeof(OPT_BASE_STRUCT),
        .priv_defaults = &(const OPT_BASE_STRUCT) {
            .min_playback_rate = 0.25,
            .max_playback_rate = 4.0,
            .ola_window_size_ms = 20,
            .wsola_search_interval_ms = 30,
        },
        .options = (const struct m_option[]) {
            {"search-interval", 
                OPT_FLOAT(wsola_search_interval_ms), M_RANGE(1, 1000)},
            {"window-size",
                OPT_FLOAT(ola_window_size_ms), M_RANGE(1, 1000)},
            {"min-speed",
                OPT_FLOAT(min_playback_rate), M_RANGE(0, FLT_MAX)},
            {"max-speed",
                OPT_FLOAT(max_playback_rate), M_RANGE(0, FLT_MAX)},
            {0}
        }
    },
    .create = af_scaletempo2_create,
};
