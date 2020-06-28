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

struct audio_bus {
    int frames;
    uint8_t **channel_data;
    int channel_count;
};

static float *audio_bus_channel(struct audio_bus *a, int ch)
{
    return (float*) a->channel_data[ch];
}

static struct audio_bus *audio_bus_new(int channels, int frames)
{
    struct audio_bus *b = malloc(sizeof(struct audio_bus));
    b->channel_count = channels;
    b->frames = frames;
    b->channel_data = malloc(channels * sizeof(uint8_t *));
    for (int i = 0; i < channels; ++i) {
        b->channel_data[i] = malloc(frames * sizeof(float));
    }
    return b;
}

static void audio_bus_free(struct audio_bus *b)
{
    if (!b) return;
    for (int i = 0; i < b->channel_count; ++i) {
        free(b->channel_data[i]);
    }
    free(b->channel_data);
    free(b);
}

static void check_overflow(int start_frame, int frames, int total_frames)
{
    assert(start_frame >= 0);
    assert(frames >= 0);
    assert(total_frames > 0);
    int sum = start_frame + frames;
    assert(sum <= total_frames);
    assert(sum >= 0);
}

static void audio_bus_zero_frames_partial(
    struct audio_bus *a, int start_frame, int frames)
{
    check_overflow(start_frame, frames, a->frames);

    if (frames <= 0)
        return;

    for (int i = 0; i < a->channel_count; ++i) {
        memset(audio_bus_channel(a, i) + start_frame, 0,
               frames * sizeof(float));
    }
}

static void audio_bus_zero_frames(struct audio_bus *a, int frames)
{
    audio_bus_zero_frames_partial(a, 0, frames);
}

static void audio_bus_zero(struct audio_bus *a)
{
    audio_bus_zero_frames(a, a->frames);
}

static void audio_bus_copy_partial_frames_to(struct audio_bus *a,
    int source_start_frame, int frame_count,
    int dest_start_frame, struct audio_bus *dest)
{
    assert(a->channel_count == dest->channel_count);
    assert(source_start_frame + frame_count <= a->frames);
    assert(dest_start_frame + frame_count <= dest->frames);

    for (int i = 0; i < a->channel_count; ++i) {
        memcpy(audio_bus_channel(dest, i) + dest_start_frame,
               audio_bus_channel(a, i) + source_start_frame,
               sizeof(float) * frame_count);
    }
}

struct audio_buffer {
    double timestamp;
    double duration;
    int frame_count;
    bool end_of_stream;
    int channel_count;
    uint8_t **channel_data;
    struct audio_buffer *next;
};

static void audio_buffer_read_frames(
    struct audio_buffer *b, int frames_to_copy,
    int source_frame_offset, int dest_frame_offset, struct audio_bus *dest)
{
    assert(!b->end_of_stream);
    assert(dest->channel_count == b->channel_count);
    assert(source_frame_offset + frames_to_copy <= b->frame_count);
    assert(dest_frame_offset + frames_to_copy <= dest->frames);
    
    // Move the start past any frames that have been trimmed.
    // source_frame_offset += trim_start_;
    
    if (!b->channel_data) {
        audio_bus_zero_frames_partial(dest, dest_frame_offset, frames_to_copy);
        return;
    }
    
    for (int ch = 0; ch < b->channel_count; ++ch) {
        const float* source_data = ((float*) b->channel_data[ch]) 
            + source_frame_offset;
        memcpy(audio_bus_channel(dest, ch) + dest_frame_offset, source_data, 
            sizeof(float) * frames_to_copy);
    }
}

static struct audio_buffer *audio_buffer_new(
    double timestamp, double duration, int frame_count, bool end_of_stream,
    int channel_count, const float **channel_data)
{
    struct audio_buffer *b = malloc(sizeof(struct audio_buffer));
    b->timestamp = timestamp;
    b->duration = duration;
    b->end_of_stream = end_of_stream;
    b->channel_count = channel_count;
    b->next = NULL;
    b->frame_count = frame_count;
    b->channel_data = malloc(sizeof(uint8_t*) * channel_count);
    int sz = frame_count * sizeof(float);
    for (int i = 0; i < channel_count; ++i) {
        b->channel_data[i] = malloc(sz);
        memcpy(b->channel_data[i], channel_data[i], sz);
    }
    return b;
}

static void audio_buffer_free(struct audio_buffer *b)
{
    for (int i = 0; i < b->channel_count; ++i) {
        free(b->channel_data[i]);
    }
    free(b->channel_data);
    free(b);
}

struct audio_buffer_queue {
    double current_time;
    int frames;
    int current_buffer_offset;
    struct audio_buffer *current_buffer;
};

static struct audio_buffer_queue *audio_buffer_queue_new(void)
{
    struct audio_buffer_queue *q = calloc(1, sizeof(struct audio_buffer_queue));
    q->current_time = MP_NOPTS_VALUE;
    // q->current_time = 0;
    // q->frames = 0;
    // q->current_buffer = NULL;
    // q->current_buffer_offset = 0;
    return q;
}

static void audio_buffer_queue_append(
    struct audio_buffer_queue *q, struct audio_buffer *buffer)
{
    if (!q->current_buffer) {
        if (buffer->timestamp != MP_NOPTS_VALUE){
            q->current_time = buffer->timestamp;
        }
        
        q->current_buffer = buffer;
    } else {
        struct audio_buffer *it = q->current_buffer;
        while (it->next) {
            it = it->next;
        }
        it->next = buffer;
    }
    q->frames += buffer->frame_count;
    assert(q->frames > 0);
}

static void audio_buffer_queue_update_current_time(
    struct audio_buffer_queue *q, struct audio_buffer *buffer, int offset)
{
    if (buffer && buffer->timestamp != MP_NOPTS_VALUE)  {
        q->current_time = buffer->timestamp
            + buffer->duration * offset / buffer->frame_count;
    }
}

static int audio_buffer_queue_internal_read(
    struct audio_buffer_queue *q, int frames,
    bool advance_position, int source_frame_offset,
    int dest_frame_offset, struct audio_bus *dest)
{
    int taken = 0;
    int current_buffer_offset = q->current_buffer_offset;
    struct audio_buffer *current_buffer = q->current_buffer;

    int frames_to_skip = source_frame_offset;
    while (taken < frames) {
        // |current_buffer| is valid since the first time this buffer is appended
        // with data. Make sure there is data to be processed.
        if (!current_buffer)
            break;

        int remaining_frames_in_buffer = current_buffer->frame_count 
            - current_buffer_offset;

        if (frames_to_skip > 0) {
            // If there are frames to skip, do it first. May need to skip into
            // subsequent buffers.
            int skipped = MPMIN(remaining_frames_in_buffer, frames_to_skip);
            current_buffer_offset += skipped;
            frames_to_skip -= skipped;
        } else {
            // Find the right amount to copy from the current buffer.
            // We shall copy no more than |frames| frames in total and 
            // each single step copies no more than the current buffer size.
            int copied = MPMIN(frames - taken, remaining_frames_in_buffer);

            // if |dest| is NULL, there's no need to copy.
            if (dest) {
                audio_buffer_read_frames(current_buffer,
                    copied, current_buffer_offset, dest_frame_offset + taken, dest);
            }

            // Increase total number of frames copied, which regulates when to end
            // this loop.
            taken += copied;

            // We have read |copied| frames from the current buffer. Advance the
            // offset.
            current_buffer_offset += copied;
        }

        // Has the buffer has been consumed?
        if (current_buffer_offset == current_buffer->frame_count) {
            if (advance_position) {
                // Next buffer may not have timestamp, so we need to update current
                // timestamp before switching to the next buffer.
                audio_buffer_queue_update_current_time(q, 
                    current_buffer, current_buffer_offset);
            }

            // If we are at the last buffer, no more data to be copied, so stop.
            if (!current_buffer->next) {
                break;
            }

            // Advances the iterator.
            current_buffer = current_buffer -> next;
            current_buffer_offset = 0;
        }
    }

    if (advance_position) {
        // Update the appropriate values since |taken| frames have been copied out.
        q->frames -= taken;
        assert(q->frames >= 0);
        assert(q->current_buffer != NULL || q->frames == 0);

        audio_buffer_queue_update_current_time(q,
            current_buffer, current_buffer_offset);

        // Remove any buffers before the current buffer as there is no going
        // backwards.
        while(q->current_buffer != current_buffer) {
            struct audio_buffer *tmp = q->current_buffer;
            q->current_buffer = q->current_buffer->next;
            audio_buffer_free(tmp);
        }
        q->current_buffer_offset = current_buffer_offset;
    }

    return taken;
}

static void audio_buffer_queue_seek_frames(struct audio_buffer_queue *q, int frames)
{
    // Perform seek only if we have enough bytes in the queue.
    assert(frames <= q->frames);
    int taken = audio_buffer_queue_internal_read(q, frames, true, 0, 0, NULL);
    assert(taken == frames);
}

static int audio_buffer_queue_read_frames(
    struct audio_buffer_queue *q, int frames,
    int dest_frame_offset, struct audio_bus *dest)
{
    assert(dest->frames >= frames + dest_frame_offset);
    return audio_buffer_queue_internal_read(q, 
        frames, true, 0, dest_frame_offset, dest);
}

static int audio_buffer_queue_peek_frames(
    struct audio_buffer_queue *q, int frames,
    int source_frame_offset, int dest_frame_offset, struct audio_bus *dest)
{
    assert(dest->frames >= frames);
    return audio_buffer_queue_internal_read(q,
        frames, false, source_frame_offset, dest_frame_offset, dest);
}

static void audio_buffer_queue_clear(struct audio_buffer_queue *q)
{
    q->current_buffer_offset = 0;
    q->frames = 0;
    q->current_time = 0;
    while(q->current_buffer) {
        struct audio_buffer *b = q->current_buffer;
        q->current_buffer = b->next;
        audio_buffer_free(b);
    }
}

static void audio_buffer_queue_free(struct audio_buffer_queue *q)
{
    if (!q) return;
    audio_buffer_queue_clear(q);
    free(q);
}

// Energies of sliding windows of channels are interleaved.
// The number windows is |input->frames()| - (|frames_per_window| - 1), hence,
// the method assumes |energy| must be, at least, of size
// (|input->frames()| - (|frames_per_window| - 1)) * |input->channels()|.
static void multi_channel_moving_block_energies(struct audio_bus *input,
    int frames_per_block, float *energy)
{
    int num_blocks = input->frames - (frames_per_block - 1);
    int channels = input->channel_count;

    for (int k = 0; k < input->channel_count; ++k) {
        const float* input_channel = audio_bus_channel(input, k);

        energy[k] = 0;

        // First block of channel |k|.
        for (int m = 0; m < frames_per_block; ++m) {
            energy[k] += input_channel[m] * input_channel[m];
        }

        const float* slide_out = input_channel;
        const float* slide_in = input_channel + frames_per_block;
        for (int n = 1; n < num_blocks; ++n, ++slide_in, ++slide_out) {
            energy[k + n * channels] = energy[k + (n - 1) * channels] - *slide_out *
                *slide_out + *slide_in * *slide_in;
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
    struct audio_bus *a, int frame_offset_a,
    struct audio_bus *b, int frame_offset_b,
    int num_frames, float *dot_product)
{
    assert(a->channel_count == b->channel_count);
    assert(frame_offset_a >= 0);
    assert(frame_offset_b >= 0);
    
    assert(frame_offset_a + num_frames <= a->frames);
    assert(frame_offset_b + num_frames <= b->frames);

    memset(dot_product, 0, sizeof(*dot_product) * a->channel_count);
    for (int k = 0; k < a->channel_count; ++k) {
        const float* ch_a = audio_bus_channel(a, k) + frame_offset_a;
        const float* ch_b = audio_bus_channel(b, k) + frame_offset_b;
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
    struct audio_bus *target_block, struct audio_bus *search_segment,
    const float *energy_target_block, const float *energy_candidate_blocks)
{
    int channels = search_segment->channel_count;
    int block_size = target_block->frames;
    int num_candidate_blocks = search_segment->frames - (block_size - 1);
    float *dot_prod = malloc(sizeof(float) * channels);
    float similarity[3];  // Three elements for cubic interpolation.

    int n = 0;
    multi_channel_dot_product(target_block, 0, search_segment, n, 
        block_size, dot_prod);
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

    multi_channel_dot_product(target_block, 0, search_segment, n, 
        block_size, dot_prod);
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
        multi_channel_dot_product(target_block, 0, search_segment, n, 
            block_size, dot_prod);

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
    free(dot_prod);
    return optimal_index;
}

// Search [|low_limit|, |high_limit|] of |search_segment| to find a block that
// is most similar to |target_block|. |energy_target_block| is the energy of the
// |target_block|. |energy_candidate_blocks| is the energy of all blocks within
// |search_block|.
static int full_search(
        int low_limit, int high_limit,
        struct interval exclude_interval,
        struct audio_bus *target_block,
        struct audio_bus *search_block,
        const float* energy_target_block,
        const float* energy_candidate_blocks)
{
    int channels = search_block->channel_count;
    int block_size = target_block->frames;
    float *dot_prod = malloc(sizeof(float) * channels);
    assert(dot_prod);

    float best_similarity = -FLT_MAX;//FLT_MIN;
    int optimal_index = 0;

    for (int n = low_limit; n <= high_limit; ++n) {
        if (in_interval(n, exclude_interval)) {
            continue;
        }
        multi_channel_dot_product(target_block, 0, search_block, n, 
            block_size, dot_prod);

        float similarity = multi_channel_similarity_measure(
            dot_prod, energy_target_block,
            &energy_candidate_blocks[n * channels], channels);

        // assert(similarity >= 0);
        
        if (similarity > best_similarity) {
            best_similarity = similarity;
            optimal_index = n;
        }
    }
    
    free(dot_prod);

    return optimal_index;
}

// Find the index of the block, within |search_block|, that is most similar
// to |target_block|. Obviously, the returned index is w.r.t. |search_block|.
// |exclude_interval| is an interval that is excluded from the search.
static int compute_optimal_index(
        struct audio_bus *search_block, struct audio_bus *target_block,
        struct interval exclude_interval)
{
    int channels = search_block->channel_count;
    assert(channels == target_block->channel_count);
    int target_size = target_block->frames;
    int num_candidate_blocks = search_block->frames - (target_size - 1);

    // This is a compromise between complexity reduction and search accuracy. I
    // don't have a proof that down sample of order 5 is optimal.
    // One can compute a decimation factor that minimizes complexity given 
    // the size of |search_block| and |target_block|. However, my experiments
    // show the rate of missing the optimal index is significant.
    // This value is chosen heuristically based on experiments.
    const int kSearchDecimation = 5;

    float *energy_target_block = malloc(sizeof(float) * channels);
    float *energy_candidate_blocks = malloc(
        sizeof(float) * channels * num_candidate_blocks);

    // Energy of all candid frames.
    multi_channel_moving_block_energies(search_block, target_size,
        energy_candidate_blocks);

    // Energy of target frame.
    multi_channel_dot_product(target_block, 0, target_block, 0,
                              target_size, energy_target_block);

    int optimal_index = decimated_search(kSearchDecimation,
                                         exclude_interval, target_block,
                                         search_block, energy_target_block,
                                         energy_candidate_blocks);

    int lim_low = MPMAX(0, optimal_index - kSearchDecimation);
    int lim_high = MPMIN(num_candidate_blocks - 1,
                            optimal_index + kSearchDecimation);
    int ret = full_search(
                lim_low, lim_high, exclude_interval, target_block,
                search_block, energy_target_block, energy_candidate_blocks);
    free(energy_target_block);
    free(energy_candidate_blocks);
    return ret;
}


// Waveform Similarity Overlap-and-add (WSOLA).
//
// One WSOLA iteration
//
// 1) Extract |target_block_| as input frames at indices
//    [|target_block_index_|, |target_block_index_| + |ola_window_size_|).
//    Note that |target_block_| is the "natural" continuation of the output.
//
// 2) Extract |search_block_| as input frames at indices
//    [|search_block_index_|,
//     |search_block_index_| + |num_candidate_blocks_| + |ola_window_size_|).
//
// 3) Find a block within the |search_block_| that is most similar
//    to |target_block_|. Let |optimal_index| be the index of such block and
//    write it to |optimal_block_|.
//
// 4) Update:
//    |optimal_block_| = |transition_window_| * |target_block_| +
//    (1 - |transition_window_|) * |optimal_block_|.
//
// 5) Overlap-and-add |optimal_block_| to the |wsola_output_|.
//
// 6) Update:
//    |target_block_| = |optimal_index| + |ola_window_size_| / 2.
//    |output_index_| = |output_index_| + |ola_window_size_| / 2,
//    |search_block_center_offset_| = |output_index_| * |playback_rate|, and
//    |search_block_index_| = |search_block_center_offset_| -
//        |search_block_center_offset_|.
// Max/min supported playback rates for fast/slow audio. Audio outside of these
// ranges are muted.
// Audio at these speeds would sound better under a frequency domain algorithm.
static const float min_playback_rate = 0.5f;
static const float max_playback_rate = 4.0f;
// Overlap-and-add window size in milliseconds.
static const int ola_window_size_ms = 20;
// Size of search interval in milliseconds. The search interval is
// [-delta delta] around |output_index_| * |playback_rate|. So the search
// interval is 2 * delta.
static const int wsola_search_interval_ms = 30;
// The maximum size in seconds for the |audio_buffer_|. Arbitrarily determined.
// static const int max_capacity_in_seconds = 3;
// The starting size in frames for |audio_buffer_|. Previous usage maintained a
// queue of 16 AudioBuffers, each of 512 frames. This worked well, so we
// maintain this number of frames.
static const int starting_buffer_size_in_frames = 16 * 512;

struct priv {
    struct mp_pin *in_pin;
    struct mp_aframe *cur_format;
    struct mp_aframe_pool *out_pool;

    bool sent_final;
    struct mp_aframe *pending;

    bool initialized;

    double speed;

    // Number of channels in audio stream.
    int channels;
    // Sample rate of audio stream.
    int samples_per_second;
    // If muted, keep track of partial frames that should have been skipped over.
    double muted_partial_frame;
    // How many frames to have in the queue before we report the queue is full.
    int capacity;
    // Book keeping of the current time of generated audio, in frames. This
    // should be appropriately updated when out samples are generated, regardless
    // of whether we push samples out when FillBuffer() is called or we store
    // audio in |wsola_output_| for the subsequent calls to FillBuffer().
    // Furthermore, if samples from |audio_buffer_| are evicted then this
    // member variable should be updated based on |playback_rate_|.
    // Note that this member should be updated ONLY by calling UpdateOutputTime(),
    // so that |search_block_index_| is update accordingly.
    double output_time;
    // The offset of the center frame of |search_block_| w.r.t. its first frame.
    int search_block_center_offset;
    // Index of the beginning of the |search_block_|, in frames.
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
    // Number of frames in |wsola_output_| that overlap-and-add is completed for
    // them and can be copied to output if FillBuffer() is called. It also
    // specifies the index where the next WSOLA window has to overlap-and-add.
    int num_complete_frames;
    // Overlap-and-add window.
    float *ola_window;
    // Transition window, used to update |optimal_block_| by a weighted sum of
    // |optimal_block_| and |target_block_|.
    float *transition_window;
    // This stores a part of the output that is created but couldn't be rendered.
    // Output is generated frame-by-frame which at some point might exceed the
    // number of requested samples. Furthermore, due to overlap-and-add,
    // the last half-window of the output is incomplete, which is stored in this
    // buffer.
    struct audio_bus *wsola_output;
    // Auxiliary variables to avoid allocation in every iteration.
    // Stores the optimal block in every iteration. This is the most
    // similar block to |target_block_| within |search_block_| and it is
    // overlap-and-added to |wsola_output_|.
    struct audio_bus *optimal_block;
    // A block of data that search is performed over to find the |optimal_block_|.
    struct audio_bus *search_block;
    // Stores the target block, denoted as |target| above. |search_block_| is
    // searched for a block (|optimal_block_|) that is most similar to
    // |target_block_|.
    struct audio_bus *target_block;
    // Buffered audio data.
    struct audio_buffer_queue *audio_buffer;
};

static int audio_renderer_algorithm_write_completed_frames_to(
    struct priv *p, int requested_frames, int dest_offset, struct audio_bus *dest)
{
    int rendered_frames = MPMIN(p->num_complete_frames, requested_frames);

    if (rendered_frames == 0)
        return 0;  // There is nothing to read from |wsola_output_|, return.

    audio_bus_copy_partial_frames_to(p->wsola_output, 0, rendered_frames,
        dest_offset, dest);

    // Remove the frames which are read.
    int frames_to_move = p->wsola_output->frames - rendered_frames;
    for (int k = 0; k < p->channels; ++k) {
        float* ch = audio_bus_channel(p->wsola_output, k);
        memmove(ch, &ch[rendered_frames], sizeof(*ch) * frames_to_move);
    }
    p->num_complete_frames -= rendered_frames;
    return rendered_frames;
}

static bool audio_renderer_algorithm_can_perform_wsola(struct priv *p)
{
    const int search_block_size = p->num_candidate_blocks
        + (p->ola_window_size - 1);
    const int frames = p->audio_buffer->frames;
    return p->target_block_index + p->ola_window_size <= frames
        && p->search_block_index + search_block_size <= frames;
}

static int audio_renderer_algorithm_frames_needed(struct priv *p)
{
    const int search_block_size = p->num_candidate_blocks
        + (p->ola_window_size - 1);
    const int frames = p->audio_buffer->frames;
    return MPMAX(frames - p->target_block_index + p->ola_window_size,
                 frames - p->search_block_index + search_block_size);
}

static bool audio_renderer_algorithm_target_is_within_search_region(struct priv *p)
{
    const int search_block_size = p->num_candidate_blocks + (p->ola_window_size - 1);

    return p->target_block_index >= p->search_block_index
        && p->target_block_index + p->ola_window_size
            <= p->search_block_index + search_block_size;
}


static void audio_renderer_algorithm_peek_audio_with_zero_prepend(
    struct priv *p, int read_offset_frames, struct audio_bus *dest)
{
    assert(read_offset_frames + dest->frames <= p->audio_buffer->frames);

    int write_offset = 0;
    int num_frames_to_read = dest->frames;
    if (read_offset_frames < 0) {
        int num_zero_frames_appended = MPMIN(
            -read_offset_frames,num_frames_to_read);
        read_offset_frames = 0;
        num_frames_to_read -= num_zero_frames_appended;
        write_offset = num_zero_frames_appended;
        audio_bus_zero_frames(dest, num_zero_frames_appended);
    }
    audio_buffer_queue_peek_frames(p->audio_buffer, num_frames_to_read,
        read_offset_frames, write_offset, dest);
}

static void audio_renderer_algorithm_get_optimal_block(struct priv *p)
{
    int optimal_index = 0;

    // An interval around last optimal block which is excluded from the search.
    // This is to reduce the buzzy sound. The number 160 is rather arbitrary and
    // derived heuristically.
    const int exclude_interval_length_frames = 160;
    if (audio_renderer_algorithm_target_is_within_search_region(p)) {
        optimal_index = p->target_block_index;
        audio_renderer_algorithm_peek_audio_with_zero_prepend(p,
            optimal_index, p->optimal_block);
    } else {
        audio_renderer_algorithm_peek_audio_with_zero_prepend(p,
            p->target_block_index, p->target_block);
        audio_renderer_algorithm_peek_audio_with_zero_prepend(p,
            p->search_block_index, p->search_block);
        int last_optimal = p->target_block_index 
            - p->ola_hop_size - p->search_block_index;
        struct interval exclude_iterval = {
            .lo = last_optimal - exclude_interval_length_frames / 2,
            .hi = last_optimal + exclude_interval_length_frames / 2
        };

        // |optimal_index| is in frames and it is relative to the beginning of the
        // |search_block_|.
        optimal_index = compute_optimal_index(p->search_block, p->target_block,
            exclude_iterval);

        // Translate |index| w.r.t. the beginning of |audio_buffer_| and extract the
        // optimal block.
        optimal_index += p->search_block_index;
        audio_renderer_algorithm_peek_audio_with_zero_prepend(p, optimal_index,
            p->optimal_block);

        // Make a transition from target block to the optimal block if different.
        // Target block has the best continuation to the current output.
        // Optimal block is the most similar block to the target, however, it might
        // introduce some discontinuity when over-lap-added. Therefore, we combine
        // them for a smoother transition. The length of transition window is twice
        // as that of the optimal-block which makes it like a weighting function
        // where target-block has higher weight close to zero (weight of 1 at index
        // 0) and lower weight close the end.
        for (int k = 0; k < p->channels; ++k) {
            float* ch_opt = audio_bus_channel(p->optimal_block, k);
            const float* const ch_target = audio_bus_channel(p->target_block, k);
            for (int n = 0; n < p->ola_window_size; ++n) {
                ch_opt[n] = ch_opt[n] * p->transition_window[n] 
                    + ch_target[n] * p->transition_window[p->ola_window_size + n];
            }
        }
    }

    // Next target is one hop ahead of the current optimal.
    p->target_block_index = optimal_index + p->ola_hop_size;
}

static void audio_renderer_algorithm_update_output_time(
    struct priv *p, float playback_rate, double time_change)
{
    p->output_time += time_change;
    // Center of the search region, in frames.
    const int search_block_center_index = (int)(
        p->output_time * playback_rate + 0.5);
    p->search_block_index = search_block_center_index 
        - p->search_block_center_offset;
}

static void audio_renderer_algorithm_remove_old_input_frames(
        struct priv *p, float playback_rate)
{
    const int earliest_used_index = MPMIN(
        p->target_block_index, p->search_block_index);
    if (earliest_used_index <= 0)
        return;  // Nothing to remove.

    // Remove frames from input and adjust indices accordingly.
    audio_buffer_queue_seek_frames(p->audio_buffer, earliest_used_index);
    p->target_block_index -= earliest_used_index;

    // Adjust output index.
    double output_time_change = ((double) earliest_used_index) / playback_rate;
    assert(p->output_time >= output_time_change);
    audio_renderer_algorithm_update_output_time(p,
        playback_rate, -output_time_change);
}

static bool audio_renderer_algorithm_run_one_wsola_iteration(
    struct priv *p, float playback_rate)
{
    if (!audio_renderer_algorithm_can_perform_wsola(p)){
        return false;
    }
    
    audio_renderer_algorithm_get_optimal_block(p);

    // Overlap-and-add.
    for (int k = 0; k < p->channels; ++k) {
        const float* const ch_opt_frame = 
            audio_bus_channel(p->optimal_block, k);
        float* ch_output = audio_bus_channel(p->wsola_output, k)
             + p->num_complete_frames;
        for (int n = 0; n < p->ola_hop_size; ++n) {
            ch_output[n] = ch_output[n] * p->ola_window[p->ola_hop_size + n] +
                ch_opt_frame[n] * p->ola_window[n];
        }

        // Copy the second half to the output.
        memcpy(&ch_output[p->ola_hop_size], &ch_opt_frame[p->ola_hop_size],
               sizeof(*ch_opt_frame) * p->ola_hop_size);
    }

    p->num_complete_frames += p->ola_hop_size;
    audio_renderer_algorithm_update_output_time(p,
        playback_rate, p->ola_hop_size);
    audio_renderer_algorithm_remove_old_input_frames(p, playback_rate);
    return true;
}

static int audio_renderer_algorithm_fill_buffer(
    struct priv *p, struct audio_bus *dest,
    int requested_frames, float playback_rate)
{
    if (playback_rate == 0)
        return 0;

    assert(p->channels == dest->channel_count);
    
    // Optimize the muted case to issue a single clear instead of performing
    // the full crossfade and clearing each crossfaded frame.
    if (playback_rate < min_playback_rate || playback_rate > max_playback_rate) {
        int frames_to_render = MPMIN(requested_frames, 
            (int) (p->audio_buffer->frames / playback_rate));

        // Compute accurate number of frames to actually skip in the source data.
        // Includes the leftover partial frame from last request. However, we can
        // only skip over complete frames, so a partial frame may remain for next
        // time.
        p->muted_partial_frame += frames_to_render * playback_rate;
        int seek_frames = (int) (p->muted_partial_frame);
        audio_bus_zero_frames(dest, frames_to_render);
        audio_buffer_queue_seek_frames(p->audio_buffer, seek_frames);

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
        const int frames_to_copy = MPMIN(requested_frames, p->audio_buffer->frames);
        const int frames_read = audio_buffer_queue_read_frames(
            p->audio_buffer, frames_to_copy, 0, dest); 
        assert(frames_read == frames_to_copy);
        return frames_read;
    }

    int rendered_frames = 0;
    do {
        rendered_frames += audio_renderer_algorithm_write_completed_frames_to(p, 
            requested_frames - rendered_frames, rendered_frames, dest);
    } while (rendered_frames < requested_frames
             && audio_renderer_algorithm_run_one_wsola_iteration(
                 p, playback_rate));
    return rendered_frames;
}

static bool audio_renderer_algorithm_available(struct priv *p) {
    return audio_renderer_algorithm_can_perform_wsola(p) 
        || p->num_complete_frames > 0;
}

static bool init_chromium(struct mp_filter *f);
static void reset(struct mp_filter *f);

static void process(struct mp_filter *f)
{
    struct priv *p = f->priv;

    if (!mp_pin_in_needs_data(f->ppins[1]))
        return;

    while (!p->initialized || !p->pending 
            || !audio_renderer_algorithm_available(p)) {
        const float *dummy[MP_NUM_CHANNELS] = {0};
        const float **in_data = dummy;
        size_t in_samples = 0;

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
            if (!init_chromium(f))
                goto error;
        }

        bool format_change =
            p->pending && !mp_aframe_config_equals(p->pending, p->cur_format);

        if (p->pending && !format_change) {
            size_t needs = audio_renderer_algorithm_frames_needed(p);
            uint8_t **planes = mp_aframe_get_data_ro(p->pending);
            int num_planes = mp_aframe_get_planes(p->pending);
            for (int n = 0; n < num_planes; n++)
                in_data[n] = (void *)planes[n];
            in_samples = MPMIN(mp_aframe_get_size(p->pending), needs);
        }

        bool final = format_change || eof;
        if (!p->sent_final) {
            struct audio_buffer *buffer = audio_buffer_new(
                //TODO handle MP_NOPTS_VALUE?
                /*timestamp*/ mp_aframe_get_pts(p->pending),
                /*duration*/ (double)in_samples / p->speed / p->samples_per_second,
                /*frame_count*/ in_samples,
                /*end_of_stream*/ false,
                /*channel_count*/ p->channels,
                /*channel_data*/ in_data);
            audio_buffer_queue_append(p->audio_buffer, buffer);
        }
        p->sent_final |= final;

        // p->rubber_delay += in_samples;

        if (p->pending && !format_change)
            mp_aframe_skip_samples(p->pending, in_samples);

        if (audio_renderer_algorithm_available(p)) {
            if (eof)
                mp_pin_out_repeat_eof(p->in_pin); // drain more next time
        } else {
            if (eof) {
                mp_pin_in_write(f->ppins[1], MP_EOF_FRAME);
                
                //reset without reinit; TODO is this correct?
                audio_buffer_queue_clear(p->audio_buffer);
                p->output_time = 0.0;
                p->search_block_index = 0;
                p->target_block_index = 0;
                audio_bus_zero(p->wsola_output);
                p->num_complete_frames = 0;
                p->capacity = starting_buffer_size_in_frames;

                p->sent_final = false;
                return;
            } else if (format_change) {
                // go on with proper reinit on the next iteration
                p->initialized = false;
                p->sent_final = false;
            }
        }
    }

    assert(p->pending);

    if (audio_renderer_algorithm_available(p)) {
        struct mp_aframe *out = mp_aframe_new_ref(p->cur_format);
        int out_samples = p->ola_hop_size * 2;
        if (mp_aframe_pool_allocate(p->out_pool, out, out_samples) < 0) {
            talloc_free(out);
            goto error;
        }

        mp_aframe_copy_attributes(out, p->pending);

        uint8_t **planes = mp_aframe_get_data_rw(out);
        assert(planes);
        int num_planes = mp_aframe_get_planes(out);
        assert(num_planes >= p->channels); //TODO correct?

        struct audio_bus dest = {
            .frames = out_samples,
            .channel_count = p->channels,
            .channel_data = planes
        };
        mp_aframe_set_pts(out, p->audio_buffer->current_time);
        out_samples = audio_renderer_algorithm_fill_buffer(
            p, &dest, out_samples, p->speed);

        if (!out_samples) {
            mp_filter_internal_mark_progress(f); // unexpected, just try again
            talloc_free(out);
            return;
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

static bool init_chromium(struct mp_filter *f)
{
    struct priv *p = f->priv;

    assert(p->pending);

    if (mp_aframe_get_format(p->pending) != AF_FORMAT_FLOATP)
        return false;

    mp_aframe_reset(p->cur_format);
    
    audio_bus_free(p->wsola_output);
    audio_bus_free(p->optimal_block);
    audio_bus_free(p->search_block);
    audio_bus_free(p->target_block);
    audio_buffer_queue_free(p->audio_buffer);

    p->muted_partial_frame = 0;
    p->output_time = 0;
    p->search_block_center_offset = 0;
    p->search_block_index = 0;
    p->num_complete_frames = 0;
    p->channels = mp_aframe_get_channels(p->pending);

    p->samples_per_second = mp_aframe_get_rate(p->pending);
    p->num_candidate_blocks = (wsola_search_interval_ms * p->samples_per_second)
        / 1000;
    p->ola_window_size = ola_window_size_ms * p->samples_per_second / 1000;
    // Make sure window size in an even number.
    p->ola_window_size += p->ola_window_size & 1;
    p->ola_hop_size = p->ola_window_size / 2;
    // |num_candidate_blocks_| / 2 is the offset of the center of the search
    // block to the center of the first (left most) candidate block. The offset
    // of the center of a candidate block to its left most point is
    // |ola_window_size_| / 2 - 1. Note that |ola_window_size_| is even and in
    // our convention the center belongs to the left half, so we need to subtract
    // one frame to get the correct offset.
    //
    //                             Search Block
    //              <------------------------------------------->
    //
    //   |ola_window_size_| / 2 - 1
    //              <----
    //
    //             |num_candidate_blocks_| / 2
    //                   <----------------
    //                                 center
    //              X----X----------------X---------------X-----X
    //              <---------->                     <---------->
    //                Candidate      ...               Candidate
    //                   1,          ...         |num_candidate_blocks_|
    p->search_block_center_offset = p->num_candidate_blocks / 2 
        + (p->ola_window_size / 2 - 1);
    p->ola_window = realloc(p->ola_window, sizeof(float) * p->ola_window_size);
    get_symmetric_hanning_window(p->ola_window_size, p->ola_window);
    p->transition_window = realloc(p->transition_window, 
        sizeof(float) * p->ola_window_size * 2);
    get_symmetric_hanning_window(2 * p->ola_window_size, p->transition_window);

    
    p->wsola_output = audio_bus_new(p->channels, 
        p->ola_window_size + p->ola_hop_size);
    // Initialize for overlap-and-add of the first block.
    audio_bus_zero(p->wsola_output);
    
    // Auxiliary containers.
    p->optimal_block = audio_bus_new(p->channels, p->ola_window_size);
    p->search_block = audio_bus_new(p->channels,
        p->num_candidate_blocks + (p->ola_window_size - 1));
    p->target_block = audio_bus_new(p->channels, p->ola_window_size);
    
    p->initialized = true;
    p->sent_final = false;
    p->audio_buffer= audio_buffer_queue_new();
        
    mp_aframe_config_copy(p->cur_format, p->pending);

    return true;
}


static bool command(struct mp_filter *f, struct mp_filter_command *cmd)
{
    struct priv *s = f->priv;

    switch (cmd->type) {
    case MP_FILTER_COMMAND_SET_SPEED:
        s->speed = cmd->speed;
        return true;
    }

    return false;
}

static void reset(struct mp_filter *f)
{
    struct priv *p = f->priv;

      // Clear the queue of decoded packets (releasing the buffers).
    audio_buffer_queue_clear(p->audio_buffer);
    p->output_time = 0.0;
    p->search_block_index = 0;
    p->target_block_index = 0;
    audio_bus_zero(p->wsola_output);
    p->num_complete_frames = 0;

    // Reset |capacity_| so growth triggered by underflows doesn't penalize
    // seek time.
    p->capacity = starting_buffer_size_in_frames;
    
    p->initialized = false;
    
    TA_FREEP(&p->pending);
}

static void destroy(struct mp_filter *f)
{
    struct priv *p = f->priv;
    free(p->ola_window);
    free(p->transition_window);
    audio_bus_free(p->wsola_output);
    audio_bus_free(p->optimal_block);
    audio_bus_free(p->search_block);
    audio_bus_free(p->target_block);
    audio_buffer_queue_free(p->audio_buffer);
    talloc_free(p->pending);
}

static const struct mp_filter_info af_drop_filter = {
    .name = "chromium",
    .priv_size = sizeof(struct priv),
    .process = process,
    .command = command,
    .reset = reset,
    .destroy = destroy,
};

static struct mp_filter *af_chromium_create(struct mp_filter *parent, void *options)
{
    struct mp_filter *f = mp_filter_create(parent, &af_drop_filter);
    if (!f) {
        talloc_free(options);
        return NULL;
    }

    mp_filter_add_pin(f, MP_PIN_IN, "in");
    mp_filter_add_pin(f, MP_PIN_OUT, "out");

    struct priv *p = f->priv;
    p->speed = 1.0;
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

const struct mp_user_filter_entry af_chromium = {
    .desc = {
        .description = 
            "Change audio speed using Chromium's WSOLA audio rendering algorithm",
        .name = "chromium",
        .priv_size = sizeof(struct priv),
    },
    .create = af_chromium_create,
};
