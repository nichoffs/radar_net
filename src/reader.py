# reader.py

from typing import List

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message as get_message_type


class RosBagReader:
    def __init__(
        self,
        input_bag: str,
        topic_names: List[str],
        storage_id: str = "mcap",
        start_offset: int = 0,
    ):
        """
        Initializes the RosBagReader.

        Args:
            input_bag: Path to the ROS bag file.
            topic_names: List of topic names to read.
            storage_id: Storage identifier (default is "mcap").
        """
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(
            rosbag2_py.StorageOptions(uri=input_bag, storage_id=storage_id),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            ),
        )
        if topic_names:
            filter = rosbag2_py.StorageFilter(topics=topic_names)
            self.reader.set_filter(filter)
        self.topic_types = self.reader.get_all_topics_and_types()
        self.msgs_read = 0

    def topic_name2type(self, topic_name: str) -> str:
        for topic_type in self.topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"Topic {topic_name} not found in the bag.")

    def read_messages(self):
        while self.reader.has_next():
            topic, data, timestamp = self.reader.read_next()
            msg_type = get_message_type(self.topic_name2type(topic))
            msg = deserialize_message(data, msg_type)
            self.msgs_read += 1
            yield topic, msg, timestamp

    def __del__(self):
        del self.reader
