"""Independent DINO memories and time-ordered replay for configured cameras."""
import heapq

from .replay_detectors import DinoReplayDetector


class DinoCameraReplay:
    kind = 'dino'
    knn = True
    frame_count = None
    capture_visual = False
    score_kind = 'dino_distance'

    def __init__(self, config, progress):
        topics = config.get('camera_topics') or config['rynnbrain']['camera_topics']
        if not topics or len(topics) != len(set(topics)):
            raise ValueError('DINO camera_topics must be a nonempty list of distinct camera topics.')
        self.cameras = {}
        shared_extractor = None
        for topic in topics:
            progress(f'Preparing independent DINO camera: {topic}')
            detector = DinoReplayDetector(config, progress, topic=topic, extractor=shared_extractor)
            shared_extractor = detector.extractor
            self.cameras[topic] = detector
        first = next(iter(self.cameras.values()))
        self.sample_fps, self.score_source = first.sample_fps, first.score_source
        self.path = first.path.parent
        self.signature = dict(protocol='dino_independent_cameras_v1',
                              cameras={topic: detector.signature for topic, detector in self.cameras.items()})
        self.excluded = set().union(*(detector.excluded for detector in self.cameras.values()))
        self.note = ('Each DINO camera has its own reference memory, calibration, score and threshold. '
                     'Cameras share one model and run sequentially; no scores are fused. ' +
                     ' '.join(f'{topic}: {detector.note}' for topic, detector in self.cameras.items()))

    def reset(self, bag):
        from rosbag_io import list_bag_topics
        available = set(list_bag_topics(bag).topic)
        missing = set(self.cameras) - available
        if missing:
            raise ValueError('Recording lacks configured DINO camera(s): ' + ', '.join(sorted(missing)))
        for detector in self.cameras.values():
            detector.reset(bag)
        self.bag = bag
        self.start_sec = min(detector.start_sec for detector in self.cameras.values())

    def iter_frames(self):
        from rosbag_io import RosbagImageSource, iter_rosbag_image_frames
        # Keep at most one pending image per camera. Each stream has its own
        # sampling schedule and no nearest-time/future-frame substitution.
        streams, heap, previous = [], [], {}
        def advance(index, topic):
            packet = next(streams[index], None)
            if packet is not None:
                frame_id, timestamp, image = packet
                if topic in previous and timestamp <= previous[topic]:
                    raise ValueError(f'DINO timestamps must increase within each camera: {topic}')
                previous[topic] = timestamp
                image.info['dino_camera_topic'] = topic
                heapq.heappush(heap, (timestamp, index, frame_id, image))
            elif topic not in previous:
                raise ValueError(f'No eligible image frames for configured DINO camera: {topic}')
        try:
            for topic, detector in self.cameras.items():
                streams.append(iter(iter_rosbag_image_frames(RosbagImageSource(self.bag, topic),
                    sample_fps=self.sample_fps, start_sec=detector.start_sec)))
                advance(len(streams) - 1, topic)
            topics = list(self.cameras)
            while heap:
                timestamp, index, frame_id, image = heapq.heappop(heap)
                topic = topics[index]
                yield (topic, frame_id), timestamp, image
                advance(index, topic)
        finally:
            for stream in streams:
                if hasattr(stream, 'close'):
                    stream.close()

    def catalogue(self):
        return [dict(topic=topic, label=topic.strip('/').split('/')[0]) for topic in self.cameras]
