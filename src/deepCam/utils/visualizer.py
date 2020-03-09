#import modules
#basic stuff
import time
import sys
import os
import numpy as np

#matplotlib stuff
import matplotlib as mpl
mpl.use('agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#basemap
from mpl_toolkits.basemap import Basemap


class CamVisualizer(object):
    
    def __init__(self):
        
        # Create figre
        numrows = 2
        numcols = 1
        fig, axvec = plt.figure(figsize=(100*numrows,20*numcols), nrows=numrows, ncols=numcols)
        
        lats = np.linspace(-90,90,768)
        longs = np.linspace(-180,180,1152)
        self.my_map = (Basemap(projection='gall', llcrnrlat=min(lats), 
                             llcrnrlon=min(longs), urcrnrlat=max(lats), 
                             urcrnrlon=max(longs), resolution = 'i' ax=axvec[0]),
                       Basemap(projection='gall', llcrnrlat=min(lats), 
                             llcrnrlon=min(longs), urcrnrlat=max(lats), 
                             urcrnrlon=max(longs), resolution = 'i', ax=axvec[1]))
        
        #set up meshgrid
        self.xx, self.yy = np.meshgrid(longs, lats)
       
        # Create new colormap
        colors_1 = [(252-32*i,252-32*i,252-32*i,i*1/16) for i in np.linspace(0, 1, 32)]
        colors_2 = [(220-60*i,220-60*i,220,i*1/16+1/16) for i in np.linspace(0, 1, 32)]
        colors_3 = [(160-20*i,160+30*i,220,i*3/8+1/8) for i in np.linspace(0, 1, 96)]
        colors_4 = [(140+80*i,190+60*i,220+30*i,i*4/8+4/8) for i in np.linspace(0, 1, 96)]
        colors = colors_1 + colors_2 + colors_3 + colors_4

        colors = list(map(lambda c: (c[0]/256,c[1]/256,c[2]/256,c[3]), colors))
        self.my_cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', colors, N=64)

        #print once so that everything is set up
        self.my_map[0].bluemarble()
        self.my_map[0].drawcoastlines()
        self.my_map[1].bluemarble()
        self.my_map[1].drawcoastlines()
        
    
    def plot(self, input_filename, output_filename, data, prediction, label):
            
        # dissect filename
        token = os.path.basename(input_filename).replace(".h5","").split("-")
        year = token[2]
        month = token[3]
        day = token[4]
        hour = token[5]
        stream = token[6]
        # Get data
        tstart = time.time()
        data = np.roll(data,[0,int(1152/2)])
        
        # Get labels
        label = np.roll(label, [0,int(1152/2)])
        l1 = (label == 1)
        l2 = (label == 2)
        print("extract data: {}".format(time.time() - tstart))

        #pdf
        #with PdfPages(filename+'.pdf') as pdf:
        
        #get figure
        numrows = 2
        numcols = 1
        fig, axvec = plt.figure(figsize=(100*numrows,20*numcols), nrows=numrows, ncols=numcols)
        
        #label
        ax = axvec[0]
        
        #draw stuff
        self.my_map[0].bluemarble()
        self.my_map[0].drawcoastlines()
        
        # Plot data
        self.x_map, self.y_map = self.my_map[0](self.xx,self.yy)
        self.my_map[0].contourf(self.x_map, self.y_map, data, 128, vmin=0, vmax=89, cmap=self.my_cmap, levels=np.arange(0,89,2))
        
        # Plot colorbar
        cbar = self.my_map[0].colorbar(ticks=np.arange(0,89,11))
        cbar.ax.set_ylabel('Integrated Water Vapor kg $m^{-2}$',size=32)
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=28)
        
        # Draw Tropical Cyclones & Atmospheric Rivers
        tc_contour = self.my_map[0].contour(self.x_map, self.y_map, l1, [0.5], linewidths=3, colors='orange')
        ar_contour = self.my_map[0].contour(self.x_map, self.y_map, l2, [0.5], linewidths=3, colors='magenta')
        
        self.my_map[0].drawmeridians(np.arange(-180, 180, 60), labels=[0,0,0,1])
        self.my_map[0].drawparallels(np.arange(-90, 90, 30), labels =[1,0,0,0])
    
        # Plot legend and title
        lines = [tc_contour.collections[0], ar_contour.collections[0]]
        labels = ['Tropical Cyclone', "Atmospheric River"]
        ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        ax.set_title("Extreme Weather Patterns {:04d}-{:02d}-{:02d}".format(int(year), int(month), int(day)), fontdict={'fontsize': 44})
        
        #pdf.savefig(bbox_inches='tight')
        #mask_ex = plt.gcf()
        #mask_ex.savefig(filename, bbox_inches='tight')
        plt.gcf().savefig(output_filename, format="PNG", bbox_inches='tight')
        plt.clf()
        
