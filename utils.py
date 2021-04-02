def listtocsv(csv_lst, file_name):
    print("saving to " + file_name)
    with open(file_name, mode='w') as cls_file:
        cls_writer = csv.writer(cls_file, delimiter=',')
        cls_writer.writerow(csv_lst)
    print("finished saving to " + file_name)
