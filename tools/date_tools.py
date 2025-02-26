from datetime import datetime, timedelta

def removeDateRange(raw_arr, beg_datetime, end_datetime):

    new_arr = []
    
    for i, d in enumerate(raw_arr):
        
        if d < beg_datetime or d > end_datetime:

            new_arr.append(d)

    return new_arr




def findMissingDatetime(raw_arr, beg_datetime, end_datetime, delta):

    sorted_arr = sorted(raw_arr)


    current_datetime = beg_datetime
    
    raw_arr_i = 0
    missing_datetime = []

    while current_datetime < end_datetime:
       
        #print("current_datetime: ", current_datetime) 

        if raw_arr_i >= len(raw_arr):
            
            missing_datetime.append(current_datetime)
            current_datetime += delta

        elif raw_arr[raw_arr_i] == current_datetime:

            #print("good")
            raw_arr_i += 1
            current_datetime += delta

        elif raw_arr[raw_arr_i] > current_datetime:   # current_datetime is missing
            
            #print("bad")
            missing_datetime.append(current_datetime)
            current_datetime += delta

        else: # raw_arr[raw_arr_i] < current_datetime
            
            #print("keep finding")
            raw_arr_i += 1 



            
            


    return missing_datetime
        


