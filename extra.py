from utils import read_csv, show_plot 

if __name__ == '__main__':
    """Merge and plot all achivment results."""
    list_of_configs, list_of_trains = read_csv('ALE/Pong-v5')
    show_plot("nn", list_of_configs, list_of_trains)