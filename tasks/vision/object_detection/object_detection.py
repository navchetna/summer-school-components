#!/usr/bin/env python3
"""
 Copyright (C) 2018-2023 Intel Corporation

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

import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
import os
import json
import cv2
#Replace below path by your open model zoo/common/python path
sys.path.append(str(Path(__file__).resolve().parents[2] / '/home/mahesh/open_model_zoo/demos/common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / '/home/mahesh/open_model_zoo/demos/common/python/openvino/model_zoo'))

from model_api.models import DetectionModel, DetectionWithLandmarks, RESIZE_TYPES, OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter

import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage
from visualizers import ColorPalette

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

class object_detection:
    def __init__(self):
        self.OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"
        self.OPEN_MODEL_ZOO_DIR = f"{os.path.expanduser('~')}/open_model_zoo"
        self.model=None
        self.IN=None
        #self.DEVICE = DEVICE
        self.list_models_supported="models.txt"
        self.at="ssd"
        self.labels=None
        self.adapter="openvino"
        self.prob_threshold=0.5
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
        self.output="output.mp4"
        self.loop=False
        

        


    def draw_detections(self,frame, detections, palette, labels, output_transform):
        frame = output_transform.resize(frame)
        for detection in detections:
            class_id = int(detection.id)
            color = palette[class_id]
            det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
            xmin, ymin, xmax, ymax = detection.get_coords()
            xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                        (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            if isinstance(detection, DetectionWithLandmarks):
                for landmark in detection.landmarks:
                    landmark = output_transform.scale(landmark)
                    cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)
        return frame


    def print_raw_results(self,detections,labels,frame_id,total_latency,total_fps):
        #log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
        #log.debug(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
        list_object=[]
        jout='{"object":" ","Latency":" ","FPS":" "}'
        y=json.loads(jout)
        i=0;
        for detection in detections:
            xmin, ymin, xmax, ymax = detection.get_coords()
            class_id = int(detection.id)
            #print(frame_id)
            det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
            #log.debug('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                    #.format(det_label, detection.score, xmin, ymin, xmax, ymax))
            #jout='{"object":" ","confidence":" ","coordinates":" ","Latency":" ","FPS":" "}'
            #y=json.loads(jout)
            list_object.append("{"+"'name':"+str('{:^9}'.format(det_label))+","+"'confidence':"+('{:10f}'.format(detection.score))+","+"'coordinates':"+('{:4},{:4},{:4},{:4} '.format(xmin, ymin, xmax, ymax)+"}"))
            #list_object.append(str('{:^9}'.format(det_label)))
            #list_object.append('{:10f}'.format(detection.score))
            #y["coordinates"]=str(out_xmin+","+out_ymin+","+out_xmax+","+out_ymax)
            #list_object.append('{:4},{:4},{:4},{:4} '.format(xmin, ymin, xmax, ymax))

            y["Latency"]=str("{:.1f} ms".format(total_latency * 1e3))
            y["FPS"]=str("{:.1f}".format(total_fps))
            #out_file=output_file
            #f = open(out_file, "a")
            #f.write(str(y))
            #f.close()
            Stat='{"status": 0, "error":"None "}'
            Status=json.loads(Stat)
            #log.debug(y,Status)
            #log.debug(Status)
            #log.debug(list_object)
           
        y["object"]=list_object
        return y,Status
            
            


    def object_detector(self,model_path: str,model_name: str,VEDIO_PATH: str,DEVICE: str,LABEL_FILE: str):
        try:
            self.model=model_path
            self.IN=VEDIO_PATH
            self.labels=LABEL_FILE
            self.DEVICE=DEVICE
            list=self.list_models_supported
            my_file = open(list, "r") 
            data = my_file.read() 
            supported_models_list = data.split("\n") 
            my_file.close()
            #model_name=self.model.split(".")
            #modelname=model_name[1].split("/")
            if model_name not in supported_models_list:
                return {"object":" ","confidence":" ","coordinates":" ","Latency":" ","FPS":" "},{"status": 1, "error": "Model not supported"}
            if self.at != 'yolov4' and self.anchors:
                log.warning('The "--anchors" option works only for "-at==yolov4". Option will be omitted')
                msg='The "--anchors" option works only for "-at==yolov4". Option will be omitted'
            if self.at != 'yolov4' and self.masks:
                log.warning('The "--masks" option works only for "-at==yolov4". Option will be omitted')
                msg='The "--masks" option works only for "-at==yolov4". Option will be omitted'
            if self.at not in ['nanodet', 'nanodet-plus'] and self.num_classes:
                log.warning('The "--num_classes" option works only for "-at==nanodet" and "-at==nanodet-plus". Option will be omitted')
                msg='The "--num_classes" option works only for "-at==nanodet" and "-at==nanodet-plus". Option will be omitted'
            cap = open_images_capture(self.IN,self.loop)

            if self.adapter == 'openvino':
                plugin_config = get_user_config(self.DEVICE, self.num_streams, self.num_threads)
                model_adapter = OpenvinoAdapter(create_core(), self.model, device=self.DEVICE, plugin_config=plugin_config,
                                                max_num_requests=self.num_infer_requests, model_parameters = {'input_layouts': self.layout})
            elif self.adapter == 'ovms':
                model_adapter = OVMSAdapter(self.model)

            configuration = {
                'resize_type': self.resize_type,
                'mean_values': self.mean_values,
                'scale_values': self.scale_values,
                'reverse_input_channels': self.reverse_input_channels,
                'path_to_labels': self.labels,
                'confidence_threshold': self.prob_threshold,
                'input_size': self.input_size, # The CTPN specific
                'num_classes': self.num_classes, # The NanoDet and NanoDetPlus specific
            }
            model = DetectionModel.create_model(self.at, model_adapter, configuration)
            model.log_layers_info()

            detector_pipeline = AsyncPipeline(model)

            next_frame_id = 0
            next_frame_id_to_show = 0

            palette = ColorPalette(len(model.labels) if model.labels else 100)
            metrics = PerformanceMetrics()
            render_metrics = PerformanceMetrics()
            presenter = None
            output_transform = None
            video_writer = cv2.VideoWriter()

            while True:
                if detector_pipeline.callback_exceptions:
                    raise detector_pipeline.callback_exceptions[0]
                # Process all completed requests
                results = detector_pipeline.get_result(next_frame_id_to_show)
                if results:
                    objects, frame_meta = results
                    frame = frame_meta['frame']
                    start_time = frame_meta['start_time']

                    presenter.drawGraphs(frame)
                    rendering_start_time = perf_counter()
                    frame = self.draw_detections(frame, objects, palette, model.labels, output_transform)
                    render_metrics.update(rendering_start_time)
                    metrics.update(start_time, frame)
                    #metrics.log_total()
                    total_latency, total_fps = metrics.get_total()
                    if len(objects) and self.raw_output_message:
                        output,Status=self.print_raw_results(objects, model.labels, next_frame_id_to_show,total_latency,total_fps)
                        #log.debug(output)

                    if video_writer.isOpened() and (self.output_limit <= 0 or next_frame_id_to_show <= self.output_limit-1):
                        video_writer.write(frame)
                    next_frame_id_to_show += 1

                    if self.no_show is False:
                        cv2.imshow('Detection Results', frame)
                        key = cv2.waitKey(1)

                        ESC_KEY = 27
                        # Quit.
                        if key in {ord('q'), ord('Q'), ESC_KEY}:
                            break
                        presenter.handleKey(key)
                    continue

                if detector_pipeline.is_ready():
                    # Get new image/frame
                    start_time = perf_counter()
                    frame = cap.read()
                    if frame is None:
                        if next_frame_id == 0:
                            raise ValueError("Can't read an image from the input")
                            msg="Can't read an image from the input"
                        break
                    if next_frame_id == 0:
                        output_transform = OutputTransform(frame.shape[:2], self.output_resolution)
                        if self.output_resolution:
                            output_resolution = output_transform.new_resolution
                        else:
                            output_resolution = (frame.shape[1], frame.shape[0])
                        presenter = monitors.Presenter(self.utilization_monitors, 55,
                                                    (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
                        if self.output and not video_writer.open(self.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                                cap.fps(), output_resolution):
                            raise RuntimeError("Can't open video writer")
                            msg="Can't open video writer"
                    # Submit for inference
                    detector_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
                    next_frame_id += 1
                else:
                    # Wait for empty request
                    detector_pipeline.await_any()

            detector_pipeline.await_all()
            if detector_pipeline.callback_exceptions:
                raise detector_pipeline.callback_exceptions[0]
            # Process completed requests
            for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
                results = detector_pipeline.get_result(next_frame_id_to_show)
                objects, frame_meta = results
                frame = frame_meta['frame']
                start_time = frame_meta['start_time']

            

                presenter.drawGraphs(frame)
                rendering_start_time = perf_counter()
                frame = self.draw_detections(frame, objects, palette, model.labels,output_transform)
                render_metrics.update(rendering_start_time)
                metrics.update(start_time, frame)
                #metrics.log_total()
                total_latency, total_fps = metrics.get_total()
                if len(objects) and self.raw_output_message:
                    output,Status=self.print_raw_results(objects, model.labels, next_frame_id_to_show,total_latency,total_fps)
                if video_writer.isOpened() and (self.output_limit <= 0 or next_frame_id_to_show <= self.output_limit-1):
                    video_writer.write(frame)

                if not self.no_show:
                    cv2.imshow('Detection Results', frame)
                    key = cv2.waitKey(1)

                    ESC_KEY = 27
                    # Quit.
                    if key in {ord('q'), ord('Q'), ESC_KEY}:
                        break
                    presenter.handleKey(key)

            
            log_latency_per_stage(cap.reader_metrics.get_latency(),
                                detector_pipeline.preprocess_metrics.get_latency(),
                                detector_pipeline.inference_metrics.get_latency(),
                                detector_pipeline.postprocess_metrics.get_latency(),
                                render_metrics.get_latency())
            for rep in presenter.reportMeans():
                log.info(rep)
            return output,Status
        except Exception as e:
            out = '{"object":" ","confidence":" ","coordinates":" ","Latency":" ","FPS":" "}'
            Stat='{"status": 1, "error":" "}'
            output=json.loads(out)
            Status=json.loads(Stat)
            Status["error"]=e
            return output,Status

#if __name__ == '__main__':
#    tsp_obj = object_detection()
#    res = tsp_obj.object_detection_demo()
