import sys
from ml_pipeline.logging import logger

class MLPipelineException(Exception):
    def __init__(self, error_message):
        self.error_message = error_message
        _, _, exc_tb = sys.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occurred in python script name [ {0} ], Line number [ {1} ], Error message [ {2} ]".format(
            self.file_name, self.lineno, str(self.error_message)
        )
    
