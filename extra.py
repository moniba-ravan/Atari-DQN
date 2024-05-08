from utils import read_csv, show_plot 

if __name__ == '__main__':

    list_of_configs, list_of_trains = read_csv()
    # print(list_of_trains)
    show_plot("nn", list_of_configs, list_of_trains)