import sys


class Logger:
    def __init__(self, log, level, job_id):
        self.log = log
        self.level = level
        self.job_id = job_id

    def write(self, message):
        for line in message.rstrip().splitlines():
            self.log.log(self.level, line.rstrip())

        sys.__stdout__.write(message)

    def flush(self):
        pass

