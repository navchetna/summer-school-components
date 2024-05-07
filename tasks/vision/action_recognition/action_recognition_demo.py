#!/usr/bin/env python3
"""
 Copyright (c) 2020-2023 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import sys
import logging as log
import os
from argparse import ArgumentParser, SUPPRESS
from os import path
import time
from collections import deque
from openvino.runtime import Core, get_version
from itertools import cycle
import sys
sys.path.append(path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))), '/home/develop/open_model_zoo/demos/common/python'))
from collections import Counter, defaultdict, deque
from functools import partial
from itertools import islice
import logging as log
import cv2
#from action_recognition_demo.models import IEModel, DummyDecoder
#from action_recognition_demo.result_renderer import ResultRenderer
#from action_recognition_demo.steps import run_pipeline
import time
from collections import OrderedDict
from itertools import chain, cycle
from threading import Thread
from enum import Enum
from queue import Queue
import monitors
from images_capture import open_images_capture
from collections import deque
from collections import defaultdict
from contextlib import contextmanager
from math import sqrt
import numpy as np
log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)
FONT_COLOR = (255, 255, 255)
FONT_STYLE = cv2.FONT_HERSHEY_DUPLEX
FONT_SIZE = 1
TEXT_VERTICAL_INTERVAL = 45
TEXT_LEFT_MARGIN = 15
Inference_output_time=0
conf_output=0
FPS_output=0
label_output=None
output_json=[]

class MovingAverageMeter:
    def __init__(self, alpha):
        self.avg = None
        self.alpha = alpha

    def update(self, value):
        if self.avg is None:
            self.avg = value
            return
        self.avg = (1 - self.alpha) * self.avg + self.alpha * value

    def reset(self):
        self.avg = None


class AverageMeter:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.sum += value
        self.count += 1

    @property
    def avg(self):
        if self.count == 0:
            return 0
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0


class WindowAverageMeter:
    def __init__(self, window_size=10):
        self.d = deque(maxlen=window_size)

    def update(self, value):
        self.d.append(value)

    @property
    def avg(self):
        return np.mean(self.d, axis=0)

    def reset(self):
        self.d.clear()

from openvino.runtime import AsyncInferQueue


def center_crop(frame, crop_size):
    img_h, img_w, _ = frame.shape

    x0 = int(round((img_w - crop_size[0]) / 2.))
    y0 = int(round((img_h - crop_size[1]) / 2.))
    x1 = x0 + crop_size[0]
    y1 = y0 + crop_size[1]

    return frame[y0:y1, x0:x1, ...]


def adaptive_resize(frame, dst_size):
    h, w, _ = frame.shape
    scale = dst_size / min(h, w)
    ow, oh = int(w * scale), int(h * scale)

    if ow == w and oh == h:
        return frame
    return cv2.resize(frame, (ow, oh))


def preprocess_frame(frame, size=224, crop_size=224, chw_layout=True):
    frame = adaptive_resize(frame, size)
    frame = center_crop(frame, (crop_size, crop_size))
    if chw_layout:
        frame = frame.transpose((2, 0, 1))  # HWC -> CHW

    return frame


class AsyncWrapper:
    def __init__(self, ie_model, num_requests):
        self.model = ie_model
        self.num_requests = num_requests
        self._result_ready = False
        self._req_ids = cycle(range(num_requests))
        self._result_ids = cycle(range(num_requests))
        self._frames = deque(maxlen=num_requests)

    def infer(self, model_input, frame=None):
        """Schedule current model input to infer, return last result"""
        next_req_id = next(self._req_ids)
        self.model.async_infer(model_input, next_req_id)

        last_frame = self._frames[0] if self._frames else frame

        self._frames.append(frame)
        if next_req_id == self.num_requests - 1:
            self._result_ready = True

        if self._result_ready:
            result_req_id = next(self._result_ids)
            result = self.model.wait_request(result_req_id)
            return result, last_frame
        else:
            return None, None


class IEModel:
    def __init__(self, model_path, core, target_device, num_requests, model_type):
        log.info('Reading {} model {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)
        if len(self.model.inputs) != 1:
            log.error("Demo supports only models with 1 input")
            sys.exit(1)

        if len(self.model.outputs) != 1:
            log.error("Demo supports only models with 1 output")
            sys.exit(1)

        self.outputs = {}
        compiled_model = core.compile_model(self.model, target_device)
        self.output_tensor = compiled_model.outputs[0]
        self.input_name = self.model.inputs[0].get_any_name()
        self.input_shape = self.model.inputs[0].shape

        self.num_requests = num_requests
        self.infer_queue = AsyncInferQueue(compiled_model, num_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model {} is loaded to {}'.format(model_type, model_path, target_device))

    def completion_callback(self, infer_request, id):
        self.outputs[id] = infer_request.results[self.output_tensor]

    def async_infer(self, frame, req_id):
        input_data = {self.input_name: frame}
        self.infer_queue.start_async(input_data, req_id)

    def wait_request(self, req_id):
        self.infer_queue[req_id].wait()
        return self.outputs.pop(req_id, None)

    def cancel(self):
        for ireq in self.infer_queue:
            ireq.cancel()


class DummyDecoder:
    def __init__(self, num_requests=2):
        self.num_requests = num_requests
        self.requests = {}

    @staticmethod
    def _average(model_input):
        return np.mean(model_input, axis=1)

    def async_infer(self, model_input, req_id):
        self.requests[req_id] = self._average(model_input)

    def wait_request(self, req_id):
        assert req_id in self.requests
        return self.requests.pop(req_id)

    def cancel(self):
        pass

class PipelineStep:
    def __init__(self):
        self.input_queue = None
        self.output_queue = VoidQueue()
        self.working = False
        self.timers = TimerGroup()
        self.total_time = IncrementalTimer()
        self.own_time = IncrementalTimer()

        self._start_t = None
        self._thread = None

    def process(self, item):
        raise NotImplementedError

    def end(self):
        pass

    def setup(self):
        pass

    def start(self):
        if self.input_queue is None or self.output_queue is None:
            raise Exception("No input or output queue")

        if self._thread is not None:
            raise Exception("Thread is already running")
        self._thread = Thread(target=self._run)
        self._thread.start()
        self.working = True

    def join(self):
        self.input_queue.put(Signal.STOP)
        self._thread.join()
        self._thread = None
        self.working = False

    def _run(self):
        self._start_t = time.time()
        self.setup()

        self.total_time = IncrementalTimer()
        self.own_time = IncrementalTimer()

        while True:
            self.total_time.tick()
            item = self.input_queue.get()

            if self._check_output(item):
                break

            self.own_time.tick()
            output = self.process(item)
            self.own_time.tock()

            if self._check_output(output):
                break

            self.total_time.tock()
            self.input_queue.task_done()
            self.output_queue.put(output)

        self.input_queue.close()
        self.end()
        self.working = False

    def _check_output(self, item):
        if is_stop_signal(item):
            self.output_queue.put(item)
            return True
        return False


class AsyncPipeline:
    def __init__(self):
        self.steps = OrderedDict()
        self.sync_steps = OrderedDict()
        self.async_step = []

        self._void_queue = VoidQueue()
        self._last_step = None
        self._last_parallel = False

    def add_step(self, name, new_pipeline_step, max_size=100, parallel=True):
        new_pipeline_step.output_queue = self._void_queue
        if self._last_step:
            if parallel or self._last_parallel:
                queue = AsyncQueue(maxsize=max_size)
            else:
                queue = StubQueue()

            self._last_step.output_queue = queue
            new_pipeline_step.input_queue = queue
        else:
            new_pipeline_step.input_queue = self._void_queue

        if parallel:
            self.steps[name] = new_pipeline_step
        else:
            self.sync_steps[name] = new_pipeline_step
        self._last_step = new_pipeline_step
        self._last_parallel = parallel

    def run(self):
        for step in self.steps.values():
            if not step.working:
                step.start()
        self._run_sync_steps()

    def close(self):
        for step in self.steps.values():
            step.input_queue.put(Signal.STOP_IMMEDIATELY)
        for step in self.steps.values():
            step.join()

    def print_statistics(self):
        log.info("Metrics report:")
        for name, step in chain(self.sync_steps.items(), self.steps.items(), ):
            log.info("\t{} total: {}".format(name, step.total_time))
            log.info("\t{}   own: {}".format(name, step.own_time))

    def _run_sync_steps(self):
        """Run steps in main thread"""
        if not self.sync_steps:
            while not self._void_queue.finished:
                pass
            return

        for step in self.sync_steps.values():
            step.working = True
            step.setup()

        for step in cycle(self.sync_steps.values()):
            step.total_time.tick()
            item = step.input_queue.get()

            if is_stop_signal(item):
                step.input_queue.close()
                step.output_queue.put(item)
                break

            step.own_time.tick()
            output = step.process(item)
            step.own_time.tock()

            if is_stop_signal(output):
                step.input_queue.close()
                step.output_queue.put(output)
                break

            step.total_time.tock()
            step.output_queue.put(output)

        for step in self.sync_steps.values():
            step.working = False
            step.end()
class BaseQueue:
    def __init__(self):
        self.finished = False

    def put(self, item, *args):
        if item is Signal.STOP_IMMEDIATELY:
            self.finished = True

    def task_done(self):
        pass

    def clear(self):
        pass

    def close(self):
        self.finished = True


class VoidQueue(BaseQueue):
    def put(self, item, *args):
        if item is Signal.STOP_IMMEDIATELY:
            self.close()

    def get(self):
        if self.finished:
            return Signal.STOP_IMMEDIATELY


class AsyncQueue(BaseQueue):
    def __init__(self, maxsize=0):
        super().__init__()
        self._queue = Queue(maxsize=maxsize)

    def put(self, item, block=True, timeout=None):
        if self.finished:
            return
        if item is Signal.STOP_IMMEDIATELY:
            self.close()
        else:
            self._queue.put(item, block, timeout)

    def close(self):
        self.finished = True
        with self._queue.mutex:
            self._queue.queue.clear()
            self._queue.queue.append(Signal.STOP_IMMEDIATELY)
            self._queue.unfinished_tasks = 0
            self._queue.all_tasks_done.notify()
            self._queue.not_full.notify()
            self._queue.not_empty.notify()

    def get(self, block=True, timeout=None):
        if self.finished:
            return Signal.STOP_IMMEDIATELY
        return self._queue.get(block, timeout)

    def clear(self):
        while not self._queue.empty():
            self.get()
            self.task_done()

    def task_done(self):
        if self.finished:
            return
        super().task_done()


class StubQueue(BaseQueue):
    def __init__(self):
        super().__init__()
        self.item = Signal.EMPTY

    def put(self, item, *args):
        if item is Signal.STOP_IMMEDIATELY:
            self.close()
        assert self.item is Signal.EMPTY
        self.item = item

    def get(self):
        if self.finished:
            return Signal.STOP_IMMEDIATELY
        item = self.item
        self.item = Signal.EMPTY
        assert item is not Signal.EMPTY
        return item


class Signal(Enum):
    OK = 1
    STOP = 2
    STOP_IMMEDIATELY = 3
    ERROR = 4
    EMPTY = 5


def is_stop_signal(item):
    return item is Signal.STOP or item is Signal.STOP_IMMEDIATELY


class ResultRenderer:
    def __init__(self, no_show, presenter, output, limit, display_fps=True, display_confidence=True, number_of_predictions=1,
                 label_smoothing_window=30, labels=None, output_height=720):
        self.no_show = no_show
        self.presenter = presenter
        self.output = output
        self.limit = limit
        self.video_writer = cv2.VideoWriter()
        self.number_of_predictions = number_of_predictions
        self.display_confidence = display_confidence
        self.display_fps = display_fps
        self.labels = labels
        self.output_height = output_height
        self.meters = defaultdict(partial(WindowAverageMeter, 16))
        self.postprocessing = [LabelPostprocessing(n_frames=label_smoothing_window, history_size=label_smoothing_window)
                               for _ in range(number_of_predictions)]

    def update_timers(self, timers):
        inference_time = 0.0
        for key, val in timers.items():
            self.meters[key].update(val)
            inference_time += self.meters[key].avg
        return inference_time

    def render_frame(self, frame, logits, timers, frame_ind, raw_output, fps):
        inference_time = self.update_timers(timers)
        global label_output
        global Inference_output_time
        global conf_output
        global FPS_output
        global output_json
        
        if logits is not None:
            labels, probs = decode_output(logits, self.labels, top_k=self.number_of_predictions,
                                          label_postprocessing=self.postprocessing)
            if raw_output:
                log.debug("Frame # {}: {} - {:.2f}% -- {:.2f}ms".format(frame_ind, labels[0], probs[0] * 100, inference_time))
        else:
            labels = ['Preparing...']
            probs = [0.]
        
        # resize frame, keep aspect ratio
        w, h, c = frame.shape
        new_h = self.output_height
        new_w = int(h * (new_h / w))
        frame = cv2.resize(frame, (new_w, new_h))

        self.presenter.drawGraphs(frame)
        # Fill text area
        fill_area(frame, (0, 70), (700, 0), alpha=0.6, color=(0, 0, 0))

        if self.display_confidence and logits is not None:
            text_template = '{label} - {conf:.2f}%'
        else:
            text_template = '{label}'

        for i, (label, prob) in enumerate(islice(zip(labels, probs), self.number_of_predictions)):
            display_text = text_template.format(label=label, conf=prob * 100)
            text_loc = (TEXT_LEFT_MARGIN, TEXT_VERTICAL_INTERVAL * (i + 1))

            cv2.putText(frame, display_text, text_loc, FONT_STYLE, FONT_SIZE, FONT_COLOR)
            print('Action: {label}'.format(label=label))
            label_output='{label}'.format(label=label)
            conf_output='{conf:.2f}%'.format(conf=prob * 100)
            print('Confidence : {conf:.2f}%'.format(conf=prob * 100))

        if frame_ind == 0 and self.output and not self.video_writer.open(self.output,
            cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame.shape[1], frame.shape[0])):
            log.error("Can't open video writer")
            return -1

        if self.display_fps:
            fps = 1000 / (inference_time + 1e-6)
            text_loc = (TEXT_LEFT_MARGIN, TEXT_VERTICAL_INTERVAL * (len(labels) + 1))
            cv2.putText(frame, "Inference time: {:.2f}ms ({:.2f} FPS)".format(inference_time, fps),
                        text_loc, FONT_STYLE, FONT_SIZE, FONT_COLOR)
            print("Inference time: {:.2f}ms".format(inference_time))
            Inference_output_time='{:.2f}ms'.format(inference_time)
            print("FPS :{:.2f}".format(fps))
            FPS_output="{:.2f}".format(fps)
            output_json.append('{"Action ":'+label_output+","+'"Confidence":'+conf_output+','+'"FPS:"'+FPS_output+','+'"Inference Time":'+Inference_output_time+"}")

        if self.video_writer.isOpened() and (self.limit <= 0 or frame_ind <= self.limit-1):
            self.video_writer.write(frame)

        if not self.no_show:
            cv2.imshow("Action Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord('q'), ord('Q'), 27}:
                return -1
            self.presenter.handleKey(key)
        
        

class LabelPostprocessing:
    def __init__(self, n_frames=5, history_size=30):
        self.n_frames = n_frames
        self.history = deque(maxlen=history_size)
        self.prev_get = None
        self.prev_label = None

    def update(self, label):
        self.prev_label = label
        self.history.append(label)

    def get(self):
        if self.prev_get is None:
            self.prev_get = self.prev_label
            return self.prev_label

        cnt = Counter(list(self.history)[-self.n_frames:])
        if len(cnt) > 1:
            return self.prev_get
        self.prev_get = self.prev_label
        return self.prev_get


def fill_area(image, bottom_left, top_right, color=(0, 0, 0), alpha=1.):
    """Fills area with the specified color"""
    xmin, ymax = bottom_left
    xmax, ymin = top_right

    image[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :] * (1 - alpha) + np.asarray(color) * alpha
    return image


def decode_output(probs, labels, top_k=None, label_postprocessing=None):
    """Decodes top probabilities into corresponding label names"""
    top_ind = np.argsort(probs)[::-1][:top_k]

    if label_postprocessing:
        for k in range(top_k):
            label_postprocessing[k].update(top_ind[k])

        top_ind = [postproc.get() for postproc in label_postprocessing]

    decoded_labels = [labels[i] if labels else str(i) for i in top_ind]
    probs = [probs[i] for i in top_ind]
    return decoded_labels, probs
def run_pipeline(capture, model_type, model, render_fn, raw_output, seq_size=16, fps=30):
    pipeline = AsyncPipeline()
    pipeline.add_step("Data", DataStep(capture), parallel=False)

    if model_type in ('en-de', 'en-mean'):
        pipeline.add_step("Encoder", EncoderStep(model[0]), parallel=False)
        pipeline.add_step("Decoder", DecoderStep(model[1], sequence_size=seq_size), parallel=False)
    elif model_type == 'i3d-rgb':
        pipeline.add_step("I3DRGB", I3DRGBModelStep(model[0], seq_size, 256, 224), parallel=False)

    pipeline.add_step("Render", RenderStep(render_fn, raw_output, fps=fps), parallel=True)

    pipeline.run()
    pipeline.close()
    pipeline.print_statistics()


def softmax(x, axis=None):
    """Normalizes logits to get confidence values along specified axis"""
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis)


class I3DRGBModelStep(PipelineStep):
    def __init__(self, model, sequence_size, frame_size, crop_size):
        super().__init__()
        self.model = model
        assert sequence_size > 0
        self.sequence_size = sequence_size
        self.size = frame_size
        self.crop_size = crop_size
        self.input_seq = deque(maxlen = self.sequence_size)
        self.async_model = AsyncWrapper(self.model, self.model.num_requests)

    def process(self, frame):
        preprocessed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preprocessed = preprocess_frame(preprocessed, self.size, self.crop_size, chw_layout=False)
        self.input_seq.append(preprocessed)
        if len(self.input_seq) == self.sequence_size:
            input_blob = np.array(self.input_seq)
            input_blob = np.expand_dims(input_blob, axis=0)
            output, next_frame = self.async_model.infer(input_blob, frame)

            if output is None:
                return None

            return next_frame, output[0], {'i3d-rgb-model': self.own_time.last}

        return frame, None, {'i3d-rgb-model': self.own_time.last}


class DataStep(PipelineStep):
    def __init__(self, capture):
        super().__init__()
        self.cap = capture

    def setup(self):
        pass

    def process(self, item):
        frame = self.cap.read()
        if frame is None:
            return Signal.STOP
        return frame

    def end(self):
        pass


class EncoderStep(PipelineStep):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.async_model = AsyncWrapper(self.encoder, self.encoder.num_requests)

    def __del__(self):
        self.encoder.cancel()

    def process(self, frame):
        preprocessed = preprocess_frame(frame)
        preprocessed = preprocessed[np.newaxis, ...]  # add batch dimension
        embedding, frame = self.async_model.infer(preprocessed, frame)

        if embedding is None:
            return None

        return frame, embedding.reshape((1, -1)), {'encoder': self.own_time.last}


class DecoderStep(PipelineStep):
    def __init__(self, decoder, sequence_size=16):
        super().__init__()
        assert sequence_size > 0
        self.sequence_size = sequence_size
        self.decoder = decoder
        self.async_model = AsyncWrapper(self.decoder, self.decoder.num_requests)
        self._embeddings = deque(maxlen=self.sequence_size)

    def __del__(self):
        self.decoder.cancel()

    def process(self, item):
        if item is None:
            return None

        frame, embedding, timers = item
        timers['decoder'] = self.own_time.last
        self._embeddings.append(embedding)

        if len(self._embeddings) == self.sequence_size:
            decoder_input = np.concatenate(self._embeddings, axis=0)
            decoder_input = np.expand_dims(decoder_input, axis=0)

            logits, next_frame = self.async_model.infer(decoder_input, frame)

            if logits is None:
                return None

            probs = softmax(logits - np.max(logits))
            return next_frame, probs[0], timers

        return frame, None, timers




class RenderStep(PipelineStep):
    """Passes inference result to render function"""

    def __init__(self, render_fn, raw_output, fps):
        super().__init__()
        self.render = render_fn
        self.raw_output = raw_output
        self.fps = fps
        self._frames_processed = 0
        self._t0 = None
        self._render_time = MovingAverageMeter(0.9)

    def process(self, item):
        if item is None:
            return
        self._sync_time()
        render_start = time.time()
        status = self.render(*item, self._frames_processed, self.raw_output, self.fps)
        self._render_time.update(time.time() - render_start)

        self._frames_processed += 1
        if status is not None and status < 0:
            return Signal.STOP_IMMEDIATELY
        return status

    def end(self):
        cv2.destroyAllWindows()

    def _sync_time(self):
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        expected_time = self._t0 + (self._frames_processed + 1) / self.fps
        if self._render_time.avg:
            expected_time -= self._render_time.avg
        if expected_time > now:
            time.sleep(expected_time - now)
class IncrementalTimer:
    def __init__(self):
        self.start_t = None
        self.total_ms = 0
        self.last = 0
        self._sum_sq = 0
        self._times = 0

    def tick(self):
        self.start_t = time.perf_counter()

    def tock(self):
        now = time.perf_counter()
        elapsed_ms = (now - self.start_t) * 1000.

        self.total_ms += elapsed_ms
        self._sum_sq += elapsed_ms ** 2
        self._times += 1
        self.last = elapsed_ms

    @property
    def fps(self):
        return 1000 / self.avg

    @property
    def avg(self):
        """Returns average time in ms"""
        return self.total_ms / self._times

    @property
    def std(self):
        return sqrt((self._sum_sq / self._times) - self.avg ** 2)

    @contextmanager
    def time_section(self):
        self.tick()
        yield
        self.tock()

    def __repr__(self):
        if not self._times:
            return "{} ms (+/-: {}) {} fps".format(float("nan"), float("nan"), float("nan"))
        return "{:.2f}ms (+/-: {:.2f}) {:.2f}fps".format(self.avg, self.std, self.fps)


class TimerGroup:
    def __init__(self):
        self.timers = defaultdict(IncrementalTimer)

    def tick(self, timer):
        self.timers[timer].tick()

    def tock(self, timer):
        self.timers[timer].tock()

    @contextmanager
    def time_section(self, timer):
        self.tick(timer)
        yield
        self.tock(timer)

    def print_statistics(self):
        for name, timer in self.timers.items():
            print("{}: {}".format(name, timer))

class action_recognition:
    
    def __init__(self):
                self.OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"
                self.OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"
                self.IN=None
                self.device =None
                self.list_models_supported="models.txt"
                self.at=None
                self.labels=None
                self.adapter="openvino"
                self.prob_threshold=0.5
                self.m_en=None
                self.m_de=None
                self.anchors=None
                self.masks=None
                self.num_classes=None
                self.num_streams="4"
                self.num_infer_requests=0
                self.num_threads=None
                self.layout=None
                self.resize_type=None
                self.mean_values=None
                self.scale_values=None
                self.reverse_input_channels=False
                self.input_size=(300, 300)
                self.raw_output_message=True
                self.output_limit=1000
                self.no_show=True
                self.output_resolution=None
                self.utilization_monitors=None
                self.output=None
                self.loop=False
                self.allow_grow=False
                self.crop_size=(0,0)
                self.decoder_seq_size=16
                self.label_smoothing=30

    def recognize(self,model_en: str,model_de: str,model_at: str,VEDIO_PATH: str,LABEL_FILE: str,DEVICE: str):
        try:      
                self.device=DEVICE
                self.at=model_at
                self.m_en=model_en
                self.m_de=model_de
                self.IN=VEDIO_PATH
                self.labels=LABEL_FILE

                if self.labels:
                    with open(self.labels) as f:
                        labels = [line.strip() for line in f]
                else:
                    labels = None

                print(labels)

                log.info('OpenVINO Runtime')
                log.info('\tbuild: {}'.format(get_version()))
                core = Core()

                decoder_target_device = 'CPU'
                if self.device != 'CPU':
                    encoder_target_device = self.device
                else:
                    encoder_target_device = decoder_target_device

                models = [IEModel(self.m_en, core, encoder_target_device, model_type='Action Recognition Encoder',
                                num_requests=1)]

                if self.at == 'en-de':
                    if self.m_de is None:
                        raise RuntimeError('No decoder for encoder-decoder model type (-m_de) provided')
                    models.append(IEModel(self.m_de, core, decoder_target_device, model_type='Action Recognition Decoder', num_requests=2))
                    seq_size = models[1].input_shape[1]
                elif self.at == 'en-mean':
                    models.append(DummyDecoder(num_requests=2))
                    seq_size = self.decoder_seq_size
                elif self.at == 'i3d-rgb':
                    seq_size = models[0].input_shape[1]

                presenter = monitors.Presenter(self.utilization_monitors, 70)
                result_presenter = ResultRenderer(no_show=self.no_show, presenter=presenter, output=self.output, limit=self.output_limit, labels=labels,
                                                label_smoothing_window=self.label_smoothing)
                cap = open_images_capture(self.IN, self.loop)
                run_pipeline(cap, self.at, models, result_presenter.render_frame, self.raw_output_message,
                            seq_size=seq_size, fps=cap.fps())

                for rep in presenter.reportMeans():
                    log.info(rep)
                print(output_json)
                Stat='{"status": 0, "error":"None "}'
                return output_json,Stat
        except Exception as e:
            out = '{"Action":" ","confidence":" ","FPS":" ","Inference Time":" "}'
            Stat='{"status": 1, "error":" "}'
            output=json.loads(out)
            Status=json.loads(Stat)
            Status["error"]=e
            return output,Status
    




        
