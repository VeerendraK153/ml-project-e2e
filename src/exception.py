import sys
#import logging
#from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno
    else:
        file_name = "<unknown>"
        line_no = "?"
    error_message = f"Error occurred in python script name [{file_name}] line number [{line_no}] error message [{str(error)}]"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Initializes the base Exception class with the raw error message
        self.error_message = error_message_detail(error_message, error_detail)
    
    def __str__(self):
        return self.error_message


'''if __name__=="__main__":
    try:
        1/0
    except Exception as e:
        logging.info('Divide by Zero')
        raise CustomException(e,sys)'''
        
