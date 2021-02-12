# The MIT License (MIT)
#
# Copyright (c) 2018 Pyjcsx
# Modifications Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.   

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
    
    def __init__(self, plot_dir=None):
        
        # Create figre        
        self.lats = np.linspace(-90,90,768)
        self.longs = np.linspace(-180,180,1152)
        
        #set up meshgrid
        self.xx, self.yy = np.meshgrid(self.longs, self.lats)

        self.my_map = Basemap(projection='gall', llcrnrlat=min(self.lats),
                              llcrnrlon=min(self.longs), urcrnrlat=max(self.lats),
                              urcrnrlon=max(self.longs), resolution = 'i')
        self.x_map, self.y_map = self.my_map(self.xx, self.yy)
        
        # Create new colormap
        colors_1 = [(252-32*i,252-32*i,252-32*i,i*1/16) for i in np.linspace(0, 1, 32)]
        colors_2 = [(220-60*i,220-60*i,220,i*1/16+1/16) for i in np.linspace(0, 1, 32)]
        colors_3 = [(160-20*i,160+30*i,220,i*3/8+1/8) for i in np.linspace(0, 1, 96)]
        colors_4 = [(140+80*i,190+60*i,220+30*i,i*4/8+4/8) for i in np.linspace(0, 1, 96)]
        colors = colors_1 + colors_2 + colors_3 + colors_4

        colors = list(map(lambda c: (c[0]/256,c[1]/256,c[2]/256,c[3]), colors))
        self.my_cmap = mpl.colors.LinearSegmentedColormap.from_list('mycmap', colors, N=64)
        
        # set plot dir
        self.plot_dir = plot_dir
        
    
    def plot(self, input_filename, output_filename, data, prediction, label):
            
        # dissect filename
        token = os.path.basename(input_filename).replace(".h5","").split("-")
        year = token[1]
        month = token[2]
        day = token[3]
        hour = token[4]
        stream = token[5]
        
        # Get data
        data = np.roll(data,[0,int(1152/2)])

        # Get predictions
        prediction = np.roll(prediction, [0,int(1152/2)])
        p1 = (prediction == 1)
        p2 = (prediction == 2)
        
        # Get labels
        label = np.roll(label, [0,int(1152/2)])
        l1 = (label == 1)
        l2 = (label == 2)
        
        #get figure
        numrows = 2
        numcols = 1
        fig, axvec = plt.subplots(figsize=(100*numrows,20*numcols), nrows=numrows, ncols=numcols)

        #do label and predictions
        for idx,ax in enumerate(axvec):
        
            #draw stuff
            self.my_map.bluemarble(ax=ax)
            self.my_map.drawcoastlines(ax=ax)
        
            # Plot data
            self.my_map.contourf(self.x_map, self.y_map, data, 128, vmin=0., vmax=1.,
                                 cmap=self.my_cmap, levels=np.arange(0., 1., 0.02), ax=ax)
        
            # Draw Tropical Cyclones & Atmospheric Rivers
            if idx == 0:
                tc_contour = self.my_map.contour(self.x_map, self.y_map, p1, [0.5], linewidths=3, colors='orange', alpha=0.9, ax=ax)
                ar_contour = self.my_map.contour(self.x_map, self.y_map, p2, [0.5], linewidths=3, colors='magenta', alpha=0.9, ax=ax)
            else:
                tc_contour = self.my_map.contour(self.x_map, self.y_map, l1, [0.5], linewidths=3, colors='orange', alpha=0.9, ax=ax)
                ar_contour = self.my_map.contour(self.x_map, self.y_map, l2, [0.5], linewidths=3, colors='magenta', alpha=0.9, ax=ax)
        
            self.my_map.drawmeridians(np.arange(-180, 180, 60), labels=[0,0,0,1], ax=ax)
            self.my_map.drawparallels(np.arange(-90, 90, 30), labels =[1,0,0,0], ax=ax)
    
            # Plot legend and title
            lines = [tc_contour.collections[0], ar_contour.collections[0]]
            labels = ['Tropical Cyclone', "Atmospheric River"]
            ax.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

            if idx == 0:
                ax.set_title("Extreme Weather Patterns {:04d}-{:02d}-{:02d} (stream {:02d})".format(int(year), int(month), int(day), int(stream)), fontdict={'fontsize': 36})

        # save figure
        if self.plot_dir is not None:
            output_filename = os.path.join(self.plot_dir, output_filename)
        plt.gcf().savefig(output_filename, format="PNG", bbox_inches='tight')
        plt.clf()
        plt.close(fig)
