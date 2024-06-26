import os 
import datetime
MAX_AGE = 60 # in Days
MAX_SIZE_MB = 1000

def get_file_age_in_days(file_path):
    # Get the current time
    current_time = datetime.datetime.now()
    
    file_mod_time = os.path.getmtime(file_path)
    
    file_mod_datetime = datetime.datetime.fromtimestamp(file_mod_time)
    
    time_difference = current_time - file_mod_datetime
    
    age_in_days = time_difference.days
    
    return age_in_days


def get_file_size_in_mb(file_path):
    # Get the file size in bytes
    file_size_bytes = os.path.getsize(file_path)
    
    # Convert the file size to megabytes
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    return file_size_mb

def test_monitoring_1():
    """
    Tests age of model.
    """
    age= get_file_age_in_days('test/model/model.h5')
    assert age < MAX_AGE, "Please redownload or update model file because max file age is exceeded"

    
def test_monitoring_2():
    """
    Tests size of model
    """
    size = get_file_size_in_mb('test/model/model.h5')
    assert size < MAX_SIZE_MB, "Model size is too big, please make sure its weight file is instantiated correctly"