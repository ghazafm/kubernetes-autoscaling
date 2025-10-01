from datetime import datetime
from logging import Logger
from typing import Optional

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


class InfluxDB:
    def __init__(self, url: str, token: str, org: str, bucket: str, logger: Logger):
        """
        Initialize connection to InfluxDB.
        Args
            url: InfluxDB URL (e.g., "http://localhost:8086")
            token: Auth token
            org: Organization name
            bucket: Target bucket
        """
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.logger = logger

        try:
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.logger.info("Connected to InfluxDB at %s", url)
        except Exception as e:
            self.logger.error("Failed to connect to InfluxDB: %s", e)
            raise

    def write_point(
        self,
        measurement,
        tags: dict,
        fields: dict,
        timestamp: Optional[datetime] = None,
    ):
        """
        Write a single point into InfluxDB.
        Args
            measurement: Measurement name (string)
            tags: Dictionary of tags (indexed metadata)
            fields: Dictionary of fields (actual values)
            timestamp: Optional datetime object (default: now)
        """
        try:
            if not fields:
                raise ValueError("At least one field must be provided")

            point = Point(measurement)

            # Add tags
            if tags:
                for k, v in tags.items():
                    point = point.tag(k, str(v))

            # Add fields
            for k, v in fields.items():
                point = point.field(k, v)

            # Add timestamp
            if timestamp:
                point = point.time(timestamp)

            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            self.logger.info(
                "Data written: %s | tags=%s | fields=%s", measurement, tags, fields
            )

        except Exception as e:
            self.logger.error("Failed to write point: %s", e)

    def close(self):
        """Close the client connection."""
        self.client.close()
        self.logger.info("InfluxDB connection closed.")
