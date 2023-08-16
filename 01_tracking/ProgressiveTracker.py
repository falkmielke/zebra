# !/usr/bin/env python3

"""
#### Progressive Video Tracker ####

Publication:
"Progressive tracking: a novel procedure to facilitate manual digitization of videos." 
Maja Mielke, Peter Aerts, Chris Van Ginneken, Sam Van Wassenbergh, Falk Mielke (2020)
Biology Open; DOI: 10.1242/bio.055962


This tool is used to track arbitrary points in a video.
Special Feature: Progressive Tracking!
    --> Press [space] to play/pause video and follow a point of interest with the mouse cursor.
        Mouse position will be tracked continuously.
        No clicking required!

To select mode (i.e. point), press numbers 1-0; Modes are defined below:
        tracking_modes = {  0: "no tracking" \
                          , 1: "first" \
                          , 2: "second" \
                          , 3: "..." \
                          ,10: "..." \
                         }
A dictionary below defines which modes are available.

Currently, the video has to be stored as an image sequence in a folder.
Any video can easily be transformed into an image sequence via ffmpeg:
    ffmpeg -i '<file.mpg>'' -r 1/1 '<folder>/$filename%04d.png'
    (https://stackoverflow.com/questions/10957412/fastest-way-to-extract-frames-using-ffmpeg)

You can zoom or pan with the tools from the matplotlib toolbar. 

In February 2021, a simple "auto tracking" functionality was added. First tests indicate that it usually 
does NOT produce reliable results. So inspect all auto-generated data.
It works by using a dlib library "correlation tracker", ORB feature detection/matching (skimage), 
and some exit criteria that break the correlation (feature matching, Procrustes comparison of features, user cancellation).
Some auto tracking parameters can be controlled ad hoc.
This is currently in testing phase, and will be implemented in the main file if it works sufficiently well.
references:
https://www.codesofinterest.com/2018/02/track-any-object-in-video-with-dlib.html
https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_orb.html

There is also a marker detection functionality. As all of these, detection accuracy depends on point contrast. 
Some blob detection parameters can be set in the "blobdict" below 
(cf. https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html ).

Auto tracking and blob detection are experimental and provided as is.


If you recorded from multiple cameras and have corresponding image points, epipolar lines can be displayed (requires opencv). 
This requires passing a list of folders to "folder" and an image_point_file.
Example image_point_file:
    file;point;u;v
    cam1.png;1;191.34;149.28

For digitizing multiple points on a single image, pass folder(s) with just the single file and an "n_points" to get n replicates of the same image.
Tracking modes can then be used as repeated digitizations.


The following shortcuts will be useful:
    [escape]|[q]:   quit
    [F1]:           show/hide help
    [space]:        play/pause (progressive tracking)
                    [ctrl] while playing: play backwards!
    [left]/[right]: previous/next frame ([shift]: jump 10 frames)
    [home]:         jump to start
    [+]/[-]:        accelerate/decelerate playback; [shift] increases steps
    [s]/[l]:        save/load
    [shift]+[s]:    save as
    [1-9]:          tracking modes
    [down]/[up]:    previous/next tracking mode
    [0]:            playback only ("no tracking" mode)
    [i]|[o]|[p]:    zoom out | center view | zoom in
    [k]:            (de-)activate pan/zoom tool
    [j]/[l]:        shift view left/right ([shift] increase pixel step)
                    [ctrl] shift view up/down
    [a]:            attempt auto digitization
    [[]/[]]:        adjust auto ROI size
    [<]/[>]:        adjust auto ROI feature count
    [{]/[}]:        adjust feature procrustes threshold
    [.]/[pageup]:   insert keyframe and return to it

    mouse wheel will adjust playback speed.
    Click on the progress bar at the bottom to jump to a frame.

Save from time to time. Filname should be associated with the video you track.



Update 2021/03: should now work with matplotlib version 3.4


"""
__author__      = "Falk Mielke"
__date__        = 20230814



################################################################################
### User Settings                                                            ###
################################################################################
# a label for saved traces
label = 'footfalls'

# tracking modes, i.e. tracemarks
tracking_modes = {   0: "no tracking" \
                  ,  1: "snout" \
                  ,  2: "ear" \
                  ,  3: "withers" \
                  ,  4: "croup" \
                  ,  5: "shoulder" \
                  ,  6: "elbow" \
                  ,  7: "carpal" \
                  ,  8: "ankle" \
                  ,  9: "hoof" \
                 }


hide_points = [] # points of which not to show the trace (for double tracking)

max_speed = 8.0

################################################################################
### Libraries                                                                ###
################################################################################
import os as OS
import sys as SYS                       
import time as TI                       # time, for process pause and date
import atexit as EXIT                   # commands to shut down processes
import threading as TH                  # threading for trigger

import numpy as NP                      # numerics
import pandas as PD                     # data storage
import matplotlib as MP                 # plotting
MP.use("TkAgg")
import matplotlib.pyplot as MPP         # plot control
import matplotlib.widgets as MPW        # for use of Cursor 
import tkinter as TK                    # GUI operations
import tkinter.messagebox as TKM        # GUI message boxes
import tkinter.filedialog as TKF        # GUI file operations


# connecting matplotlib and tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler

coordinates = ['x','y']

# auto tracking with dlib and skimage ORB features
import dlib as DL # image landmark tools
import skimage.feature as SKIF # feature extraction
import skimage.color as SKIC # grayscale conversion
import skimage.util as SKIU # more grayscale
import skimage.exposure as SKIEXP # histogram correction




################################################################################
### General Helpers                                                          ###
################################################################################

### ProcrustesDistance

def Centroid(pointset):
    # the centroid of the data.
    return NP.mean(pointset, axis = 0)

def RCS(pointset):
    # root centroid size, i.e. the scale of the data, defined as the root sum of 
    # euclidean distances of all points from their centroid.
    return NP.sqrt(NP.sum(NP.power(pointset-Centroid(pointset), 2) ))

def Center(pointset):
    # move all points so that the centroid is at the origin.
    return pointset - Centroid(pointset)

def UnitScale(pointset):
    # Scale the points so that RCS is one.
    return pointset / RCS(pointset)

def Standardize(pointset):
    # Standardize a point set, i.e. center and scale to unity.
    return UnitScale(Center(pointset.copy()))

def ProcrustesDistance(focal_points, ref_points):
    # optimum rotation matrix to rotate focal points to reference points
    # points must be Nx3 numpy arrays.
    # following https://en.wikipedia.org/wiki/Kabsch_algorithm
    # cf. 
    #   http://nghiaho.com/?page_id=671
    #   https://github.com/charnley/rmsd
    #   http://www.kwon3d.com/theory/jkinem/rotmat.html

    if not (focal_points.shape == ref_points.shape):
        raise IOError('point sets must have the same shape. {} / {}'.format(focal_points.shape, ref_points.shape))

    # deep copy
    focal_points = focal_points.copy()
    ref_points = ref_points.copy()


    # standardize 
    focal_points = Standardize(focal_points)
    ref_points = Standardize(ref_points)


    # calculate cross-dispersion (correlation) matrix
    cross_dispersion = focal_points.T @ ref_points
    
    # singular value decomposition of cross-dispersion matrix
    U, singular_values, V = NP.linalg.svd(cross_dispersion, full_matrices = False)

    # if R has negative determinant, it is a reflection matrix and must be modified.
    if NP.linalg.det(V.T @ U.T) < 0:
        singular_values[-1] *= -1

    # return the procrustes Distance
    return 1 - (singular_values.sum())**2





RectToPosition = lambda rect: NP.array([int(rect.left()) \
                            , int(rect.top()) \
                            , int(rect.right()) \
                            , int(rect.bottom()) \
                            ])


def ColorToGray(img):
    if img.shape[-1] == 4:
        img = SKIC.rgba2rgb(img)
    return SKIC.rgb2gray(img)


def ImageTo8Bit(img):
    return NP.array(img.copy() * (2**8-1.), dtype = NP.uint8)

def ExtractRegionOfInterest(img, pos):
    if len(img.shape) == 3:
        img = img[pos[1]:pos[3], pos[0]:pos[2], :].copy()
    else:
        img = img[pos[1]:pos[3], pos[0]:pos[2]].copy()
    return img


def GetKeypoints(orb, extract):
    orb.detect_and_extract(extract)
    kp = orb.keypoints
    desc = orb.descriptors

    return kp, desc


def MatchedFeatures(orb, roi1, roi2):
    try:    
        kp1, desc1 = GetKeypoints(orb, roi1)
        kp2, desc2 = GetKeypoints(orb, roi2)

        matches = SKIF.match_descriptors(desc1, desc2, cross_check = True)

    except RuntimeError as rte:
        return [], []
    except ValueError as ve:
        return [], []
    # ax = MPP.gca()
    # SKIF.plot_matches(ax, roi1, roi2, kp1, kp2, matches)
    # MPP.show()

    return kp1[matches[:, 0]], kp2[matches[:, 1]]



# AppendZeroColumn = lambda nx_array: NP.c_[ nx_array, NP.zeros((nx_array.shape[0], 1)) ]


### Blob Detection
# works excellent on manual and progressive tracking
# but only reasonably well on auto

# https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html

# The radius of each blob is approximately 2–√σ for a 2-D image and 3–√σ for a 3-D image.
blobdict = dict(min_sigma=2, max_sigma=32, num_sigma=8, threshold=.1, overlap=0.33, exclude_border = 2) 
# counter = 0
def BlobDetect(img, center, r0 = None, frame = None, normalize = True, *bt_args, **bt_kwargs):


    if frame is None:
        window = img
        x0 = 0
        y0 = 0
    else:
        y0 = int(center[1]-frame)
        x0 = int(center[0]-frame)
        window = img[y0:y0+2*frame, x0:x0+2*frame]

    if normalize:
        p2, p98 = NP.percentile(img, (2, 98))
        window = SKIEXP.rescale_intensity(window, in_range=(p2, p98))

    # blob_dog, blob_log, blob_doh
    log_blobs = SKIF.blob_log(window, *bt_args, **bt_kwargs) # **blobdict
    log_blobs[:, 2] = log_blobs[:, 2] * NP.sqrt(2)

    if len(log_blobs) == 0:
        # print ('no blob found')
        return center, 0

    if r0 is None:
        found = NP.stack([NP.array([blob[1], blob[0], blob[2], NP.sqrt((blob[1]-window.shape[0]/2)**2 + (blob[0]-window.shape[1]/2)**2 )]) for blob in log_blobs])
    else:
        found = NP.stack([NP.array([blob[1], blob[0], blob[2], NP.sqrt((blob[1]-window.shape[0]/2)**2 + (blob[0]-window.shape[1]/2)**2  + (blob[2]-r0)**2 )]) for blob in log_blobs])

    x, y, r, d = list(found[NP.argmin(found[:, 3]), :])

    # global counter
    # counter += 1
    

    # fig = MPP.figure()
    # ax = fig.add_subplot(1,1,1)

    # ax.imshow(window, cmap = 'gray', origin = 'lower')
    # c = MPP.Circle((x, y), r, color='y', linewidth=0.5, fill=False)
    # ax.add_patch(c)
    # ax.scatter(x, y, s = 5 \
    #                         , linewidth = 0.5 \
    #                         , edgecolor = 'r' \
    #                         , facecolors = 'r' \
    #                         , marker = '+' \
    #                         , zorder = 24 \
    #                         )

    # ax.set_xlim([0, window.shape[1]])
    # ax.set_ylim([0, window.shape[0]])

    # MPP.tight_layout()
    # fig.savefig(f'debug/{counter:04.0f}.png', dpi = 300)

    # MPP.close(fig)

    return (x0+x, y0+y), r


### remove axis labels from a matplotlib axis
def FullDespine(ax, xlab = False):

    ax.get_xaxis().set_tick_params(which='both', bottom = False, top = False, labelbottom = False)
    ax.get_yaxis().set_tick_params(which='both', bottom = False, top = False, labelbottom = False)
    # ax.get_xaxis().set_tick_params(which='both', direction='out')
    # ax.get_yaxis().set_tick_params(which='both', direction='out')

    ax.tick_params(top=False)
    ax.tick_params(right=False)
    ax.tick_params(left=False)
    ax.tick_params(right=False)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if not xlab:
        ax.set_xticks([])
    ax.set_yticks([])

### colormap for the progress bar
trackstatus_color = { \
            'red':   ((0.0, 0.1, 0.2)
                     ,(1.0, 0.9, 0.0))
        ,   'green': ((0.0, 0.1, 0.5)
                     ,(1.0, 0.9, 0.0))
        ,   'blue':  ((0.0, 0.1, 0.2)
                     ,(1.0, 0.9, 0.0))
        ,   'alpha': ((0.0, 1.0, 1.0)
                     ,(1.0, 1.0, 0.0))
        }



### default font used for plotting 
the_font = {'family': 'sans' # -serif
            ,'sans-serif': ['DejaVu sans mono'] # Helvetica # Liberation Sans
            ,'weight' : 'normal' \
            ,'size'   : 2 \
            }
MP.rc( 'font', **the_font )

save_prefix = r"%s" % ("{:s}".format(TI.strftime('%Y%m%d')))




################################################################################
### Video Loading                                                            ###
################################################################################
class VideoBuffer(object):  
    """
    An object of this class holds the video folder and loads images in a background thread.
    """

#______________________________________________________________________________
### constructor
    def __init__(self, folder, n_points = 1, *args, **kwargs):

        # folder and file paths
        self.folder = folder
        self.files = list(sorted(\
                        [fi for fi in OS.listdir(self.folder) \
                            if OS.path.splitext(fi)[-1] in ['.png', '.tif', 'tiff', '.jpg', '.jpeg'] \
                        ] ))

        if len(self.files) == 0:
            raise IOError('no image files found.')

        # for digitizing many points on a static image, single image mode is useful.
        self.single_image_mode = len(self.files) == 1
        if self.single_image_mode:
            self.files = self.files * n_points


        # data shape
        test_image = self.LoadImage(0) # grab a test image to extract video info
        self.shape = NP.array((len(self.files), *[sh for sh in test_image.shape]), dtype = int)


        # data storage
        # Note: this should work with RGB images, but that was not tested. 
        # Change the "imshow" colormap for RGB display.
        self.data = NP.empty(tuple(self.shape), dtype = test_image.dtype) # a matrix storing the image data
        self.loaded = NP.zeros( (self.shape[0],), dtype = bool ) # a bool vector indicating whether an image is loaded or not
        self.loading = False # activity indicator, used to stop a buffer loading process
        self._loader = None # thread to load frames in the background
        self.pointer = 0 # current image
        self.interval = [-1500, +2500] # images loaded by buffer relative to the current frame


    def __len__(self):
        ## number of images
        return self.shape[0]



#______________________________________________________________________________
### Loading
    def LoadImage(self, nr):
        ## get a single image file; for simplicity, this uses the matplotlib function.
        # img = MPP.imread("/".join([self.folder, self.files[nr]]), 0)
        # return NP.array(img)/(2**int(str(img.dtype)[5:]))
        return MPP.imread("/".join([self.folder, self.files[nr]]))


    def FillImageBuffer(self, frames):
        ## loading a series of images; used as function in the buffer thread

        for nr, frame_nr in enumerate(frames):
            if not self.loading:
                # exit if loading is interrupted
                break

            if self.loaded[frame_nr]:
                # skip previously loaded frames
                continue

            # print('loading', frame_nr)
            # load the single image
            self.data[frame_nr] = self.LoadImage(frame_nr)
            self.loaded[frame_nr] = True # update status


    def UpdateBuffer(self):
        ## preparing and running the buffering thread

        # define a range of images to load: 
            # first current frame
            # then subsequent frames
            # then frames before the current one
        from_img = NP.max([self.pointer + self.interval[0], 0])
        to_img = NP.min([self.pointer + self.interval[1], len(self)])
        load_order = list(range(self.pointer, to_img, 1)) + list(range(self.pointer-1, from_img, -1))

        # wait for a previous buffering process to finish
        self.Interrupt()

        # start loading
        self.loading = True # activity lock
        self._loader = TH.Thread(target = self.FillImageBuffer, args = (load_order,)) # initialize background process
        self._loader.daemon = True # make this thread a daemon
        self._loader.start() # and go!



#______________________________________________________________________________
### handling active image frame
    def ChangeImgPointer(self, change):
        ## changing the currently active image by frame number difference 
        target = self.pointer + change

        # make sure the target frame is in the range of frames
        if target < 0:
            target = 0
        if target >= len(self):
            target = len(self)-1

        # adjust the pointer
        self.pointer = target
        # print ('frame', self.pointer)
        self.UpdateBuffer() # start buffering after changing the image


    def GoToFrame(self, target_frame):
        ## changing the currently active image by an absolute image number
        self.ChangeImgPointer(target_frame - self.pointer)



#______________________________________________________________________________
### image retrieval
    def __getitem__(self, nr):
        ## retrieving an image from the stored ones

        # load the image with priority if it had not been loaded before
        if not(self.loaded[nr]):
            self.Interrupt() # finish current buffering procedure
            self.LoadImage(nr) # load the selected image
            self.UpdateBuffer() # restart buffer

        # return the selected image
        return self.data[nr]


    def GetCurrent(self):
        ## get the currently active image
        return self[self.pointer]



#______________________________________________________________________________
### Threading
    def Status(self):
        ## indicate how many images are loaded
        print (NP.sum(self.loaded), '/', len(self))


    def Stop(self):
        ## interrupt the buffering procedure
        self.loading = False


    def Interrupt(self):
        ## interrupt the buffering
        self.Stop() # let the buffer stop
        if self._loader is None:
            # nothing to do if the buffering had not started
            return

        # wait for the process to finish (should be immediate if 'self.loading == False')
        self._loader.join()


################################################################################
### Help Overlay                                                             ###
################################################################################
class HelpAnnotation(object):
    # a class to manage helpful comments on the image axis
    def __init__(self, ax, position, text, color = 'w'):
        self.ax = ax
        self.position = position
        self.text = text
        self.color = color
        self.is_visible = False
        self.note = None


    def Draw(self, color, *args, **kwargs):
        # make the comment appear on the axis
        if color is None:
            color = self.color

        if self.is_visible and (self.note is not None):
            self.note.set_visible(False)
            self.is_visible = False
            return

        self.note = self.ax.annotate( \
                              self.text \
                            , xy = self.position \
                            , xycoords = 'axes fraction' \
                            , xytext = self.position \
                            , textcoords = 'axes fraction' \
                            , va = 'top', ha = 'left' \
                            , zorder = 100 \
                            , color = color \
                            , *args, **kwargs
                            )
        self.is_visible = True




################################################################################
### Epipolar Geometry                                                        ###
################################################################################
def PrepareEpipolarGeometry(image_point_file):
    # calculate epilines from image points
    # read a csv file of corresponding image points from different cameras
    # header: file;point;u;v
    # return cross camera fundamental matrices and epiline function from openCV

    # import opencv toolbox
    import cv2 as CV


    # load csv with points
    image_points = PD.read_csv(image_point_file, sep = ';')

    # better insex
    files = NP.unique(image_points['file'].values)
    image_points.set_index(['file', 'point'], inplace = True)
    
    # openCV requires float32
    image_points = image_points.astype(NP.float32)

    # dict of date per camera view
    image_points = {cam: image_points.loc[fi, :] for cam, fi in enumerate(files)}

    # align indices
    idx = image_points[0].index.values
    image_points = {cam: imp.loc[idx, :] for cam, imp in image_points.items()}    

    fundamental_matrices = {(cam1, cam2): CV.findFundamentalMat(image_points[cam1].values \
                                                            , image_points[cam2].values \
                                                            , CV.FM_LMEDS \
                                                            )[0] \
                      for cam1 in range(len(files)) for cam2 in range(len(files)) \
                     }
    EpilineCalculator = CV.computeCorrespondEpilines

    return fundamental_matrices, EpilineCalculator



################################################################################
### Point Tracker GUI                                                        ###
################################################################################
class Tracker(TK.Tk):
    """
    This class handles the tracking process.
    It constructs the tracking GUI and stores the data.
    It is connected to an image buffer that loads video data from an image series. 
    constructor arguments:
        folder ... if 'None', launches a folder selection dialog. If string, take that folder. If list, then multiple cameras are initiated.
        image_point_file ... optional; path to csv containing corresponding image points from multiple views to calculate epilines.
        n_points ... in single image mode, how many points of interest are displayed
    """

#______________________________________________________________________________
### constructor
    def __init__(self, folder = None, image_point_file = None, n_points = 1, fullscreen = False, *args, **kwargs):
        ## constructor

        # the folder that holds the video image series
        if folder is None:

            openfile = TK.Tk()
            openfile.withdraw()
            folder = TKF.askdirectory( \
                                      title = 'Select directory with image sequence for tracking.' \
                                    , parent = openfile \
                                    )
            openfile.quit()

            if (folder is None) or (folder == '') or (len(folder) == 0): 
                raise Exception('No folder selected!')
        
        # print (folder, type(folder))
        if hasattr(folder, '__iter__') and not type(folder) is str: 
            self.folder = folder
        else: 
            self.folder = [folder]

        # call the "Quit" function if the tracker exits unexpectedly
        EXIT.register(self.Quit)

        # start image loading
        self.all_buffers = [VideoBuffer(folder = fold, n_points = n_points) for fold in self.folder]
        if not NP.all(NP.diff(NP.stack([img_buffer.shape for img_buffer in self.all_buffers], axis = 0), axis = 0) == 0 ):
            raise IOError('image folders must contain the same number of image files!')
        for img_buffer in self.all_buffers: 
            img_buffer.UpdateBuffer() # start loading images

        # check if tracker is in single image mode; hide all labels
        self.single_image_mode = all([img_buffer.single_image_mode for img_buffer in self.all_buffers])
        if self.single_image_mode:
            hide_points = [key for key in tracking_modes.keys() if not (key == 0)]

        # get a GUI master
        super(Tracker, self).__init__(*args, **kwargs)

        # start in full screen
        self.attributes("-fullscreen", fullscreen)
        self.title("Progressive Video Tracker")


        # tracker properties
        self.saved = False # handle data saving 
        self.outfile = None # path where to save
        self.modifiers = {mod: False for mod in ['control', 'shift', 'alt']} # keyboard modifiers
        self.pos = (NP.nan, NP.nan) # storage of the pointer position

        self.cameras = range(len(self.folder)) # possible camera views
        self.view = 0 # which video file is viewed
        self.imagebuffer = self.all_buffers[self.view] # the currently selected buffer

        self.mode = 0 # tracking mode
        self.playing = False # progressive tracking
        self.speed = 1e-1 # tracking speed
        self.exiting = False # make sure exit is only queried once
        self.grayscale = len(self.imagebuffer.shape) < 4 # monochrome or rgb images
        self.clicking = False # record if a pan/zoom action is in progress
        self.help = [] # help annotations


        # auto tracking
        self.auto_tracking = False
        self.roi_halfedge = 32 # half edge length of auto tracking window
        self.n_features = 128 # number of features to match auto tracking progress
        self.pd_threshold = 0.48 # procrustes distance below which the thing is canceled

        self.keyframe = 0


        # list of modes
        self.modelist = list(sorted(tracking_modes.keys()))
        self.modelabels = [tracking_modes[tm] for tm in sorted([key for key in tracking_modes.keys()])]


        # data storage
        self.data = {}
        for view in self.cameras:
            self.data[view] = PD.DataFrame(columns = [ f"{tracking_modes[tm]}_{coord}" \
                                                  for tm in sorted(tracking_modes.keys()) \
                                                  for coord in coordinates \
                                                  if not (tm == 0)\
                                              ] \
                            , index = NP.arange( self.imagebuffer.shape[0] ) ) 
            self.data[view][:] = NP.NAN # initialize empty
            self.data[view].index.name = "frame_nr" # frame number as data frame index

        # prepare the GUI
        self.EpipolarGeometry(image_point_file)
        self.MakeFigure()
        self.MakeGUI()
        self.UpdatePlot(reset = True)
        self.ShowHelp()

        # begin the GUI main process
        self.mainloop()

        


#______________________________________________________________________________
### magic data access
    def __setitem__(self, key, value):
        ## changing a data entry with 'self[...] = ...'
        # key is a list of frame number (self.imagebuffer.pointer) and tracking mode (self.mode)
        # value is a list of xy coordinates
        self.data[self.view].loc[key[0], [f"{tracking_modes[key[1]]}_{coord}" for coord in coordinates]] = value
        self.saved = False


    def __getitem__(self, key):
        ## retrieving a data entry with 'self[...]'
        # key is a list of frame number (self.imagebuffer.pointer) and tracking mode (self.mode)
        return self.data[self.view].loc[key[0], [f"{tracking_modes[key[1]]}_{coord}" for coord in coordinates]].values


    def GetIndicatorText(self, note = ''):
        ## return the status text that is displayed in the Tk label widget 
            return (f'{self.folder[self.view]}, ' if len(self.cameras) > 1 else '') + f"{tracking_modes[self.mode]}, v={NP.log10(1/self.speed):.1f}, {'pt' if self.single_image_mode else 't'}={self.imagebuffer.pointer} {note}"


#______________________________________________________________________________
### keyboard shortcuts
    def PressKey(self, event):
        ## this function handles keyboard press commands by the user.

        # print('pressed', event.key, event.xdata, event.ydata, event.guiEvent)
        if event.key is None:
            # ignore if there was no keyboard key pressed (should not be relevant)
            if False and 'keysym=Tab' in str(event.guiEvent):
                event.key = 'tab'
                # self.focus_set('.!canvas')
            else:
                return

        # toggle modifier events
        if event.key in self.modifiers.keys():
            self.modifiers[event.key] = True
            return

        elif 'shift' in event.key:
            self.modifiers['shift'] = True

        elif 'ctrl' in event.key:
            self.modifiers['control'] = True


        # change the mode if the key is one of the numbers in the tracking modes
        if event.key in ['%i' % (key) for key in tracking_modes.keys()]:
            self.SetMode(int(event.key))


        # execute a keyboard shortcut, or no operation if the key is not set.
        self.keyboard_shortcuts.get(event.key.lower().replace('ctrl+', ''), self.NOOP )(event)

        


    def ReleaseKey(self, event):
        ## this function handles keyboard release commands by the user.
        # print('released', event.key, event.xdata, event.ydata)

        # modifiers released
        if event.key in self.modifiers.keys():
            self.modifiers[event.key] = False
            return

        if event.key is None:
            # for some reason, shift and control release are not captured
            self.modifiers['control'] = False
            self.modifiers['shift'] = False
            return

        if 'shift' in event.key:
            self.modifiers['shift'] = False

        elif 'ctrl' in event.key:
            self.modifiers['control'] = False


    def ClickRelease(self, _):
        # record if mouse click is released
        self.clicking = False


    def ClickCapture(self, event):
        ## this function handles what happens if the user is clicking on an axis
        # print ('click', event.name, event.key, event.button, event.step)
        # event.button in [MouseButton.LEFT  MouseButton.MIDDLE MouseButton.RIGHT]
        # 'button', 'canvas', 'dblclick', 'guiEvent', 'inaxes', 'key', 'lastevent', 'name', 'step', 'x', 'xdata', 'y', 'ydata'

        # record if mouse is clicked
        self.clicking = True

        # clicking is ignored in progressive tracking.
        if self.playing:
            return

        # a click on the progress axis will jump to a frame
        if event.inaxes == self.progress_ax:
            self.ClickTime(event)

        # a click on the image axis will track a point; unless clicking is deactivated.
        if (event.inaxes == self.image_ax) and (not self.noclick.get()):
            self.TrackPoint(event)

        ## forwarding click handling (not necessary)
        # key_press_handler(event, self._canvas, self.toolbar)


    def MousewheelCapture(self, event):
        ## this function handles what happens if the user is clicking on an axis
        # print ('wheel', dir(event))
        # print ('wheel', event.name, event.key, event.button, event.step)

        # a scroll on the progress axis will move in time
        if event.inaxes == self.progress_ax:
            if event.button == 'up':
                self.Forward()
            elif event.button == 'down':
                self.Backward()

        # a scroll on the image axis will adjust speed
        if event.inaxes == self.image_ax:
            self.ChangeSpeed(event)
            

    def TrackPoint(self, event):
        ## point tracking by clicking on the axis
        
        # do not track if a tool from the toolbar is selected
        if self.image_ax.get_navigate_mode():
            # https://github.com/matplotlib/matplotlib/issues/18148
            return

        # if a landmark is actively tracked, change the coordinate based on click position
        if not (self.mode == 0):

            xy = (event.xdata, event.ydata)
            r = None
            if self.blobdet.get():
                img = self.imagebuffer.GetCurrent()
                if not self.grayscale:
                    img = ColorToGray(img)

                xy, r = BlobDetect(SKIU.invert(img), center = xy, frame = self.roi_halfedge \
                                   , **blobdict)



            self[(self.imagebuffer.pointer, self.mode)] = xy

            # testing: draw blob
            if r is not None:
                c = MPP.Circle((xy[0], xy[1]), r, color='y', linewidth=0.5, fill=False)
                self.image_ax.add_patch(c)

                self._canvas.draw()
                self.update()

        # go to next frame (or quick progress if shift is held)
        self.ChangePointer(change = +10 if self.modifiers['shift'] else +1)


        
    def ClickTime(self, event):
        ## change current frame by clicking on the progress axis
        timeclick = int(event.xdata) # x position of the click

        # make sure a valid frame is clicked
        if timeclick < 0 or timeclick >= len(self.imagebuffer):
            return

        # change active frame
        self.imagebuffer.GoToFrame(timeclick)

        # re-draw
        self.UpdatePlot()



#______________________________________________________________________________
### diagram control
    def EmptyPlot(self, reset = False):
        ## empty the axes prior to update

        # image axis
        if not reset:
            limits = [self.image_ax.get_xlim(), self.image_ax.get_ylim()]

        self.image_ax.cla() # clear the image axes
        FullDespine(self.image_ax) # remove spines

        if reset:
            self.image_ax.set_xlim([0., self.imagebuffer.shape[2]])
            self.image_ax.set_ylim([self.imagebuffer.shape[1], 0.])
        else:
            self.image_ax.set_xlim([*limits[0]])
            self.image_ax.set_ylim([*limits[1]])


        self.progress_ax.cla() # clear progress axes
        FullDespine(self.progress_ax, xlab = True) # remove all except x axis
        
        # restore x labels on progress axis
        self.progress_ax.get_xaxis().set_tick_params(which='both', length = 0)
        self.progress_ax.set_xlim([0,self.imagebuffer.shape[0]])
        self.progress_ax.set_ylim([0,len(tracking_modes)-1])
        

    def PlotPoints(self, key, interval, color = 'w', *args, **kwargs):
        ## plot previously tracked points in a common way

        # find which frames are to be shown (in interval around current frame)
        if self.single_image_mode:
            # in single image mode, all points are shown.
            frames = self.data[self.view].index.values
        else:
            # otherwise, a subset of frames is selected.
            frames = NP.arange(NP.max([self.imagebuffer.pointer+interval[0], 0]) \
                             , NP.min([NP.max([self.imagebuffer.pointer+interval[1], 0]), len(self.imagebuffer)-1]) \
                             )

        # get the points
        points = self.data[self.view].loc[ \
                          frames \
                        , [f"{tracking_modes[key]}_{coord}" for coord in coordinates] \
                        ].values
        
        # plot the markers
        self.image_ax.scatter( \
                              points[:,0], points[:,1] \
                            , s = 5 \
                            , linewidth = 0.5 \
                            , color = color \
                            #, facecolors = color \
                            , marker = '+' \
                            , zorder = 24 \
                            , *args, **kwargs
                            )


    def PlotEpilines(self):
        # plot epipolar lines from other traces onto the current image
        

        if not self.show_epilines.get():
            # epilines are hidden
            return


        # loop all other cams
        for crosscam in self.cameras:
            if self.view == crosscam:
                continue
            
            # assemble x and y data
            x = [0, self.imagebuffer.shape[2]]
            y = self.current_epilines[(self.view, crosscam)].loc[self.imagebuffer.pointer, :].values

            # plot
            self.image_ax.plot(x, y, color = 'w', lw = 0.5, alpha = 0.5, ls = '-', zorder = 1000)



    def AutoPanOnMargin(self):
        # pan view if tracking click is close to margin
        limits = NP.array([list(self.image_ax.get_xlim()), list(self.image_ax.get_ylim())])
        extent = NP.diff(limits, axis = 1)
        px, py = self[(self.imagebuffer.pointer-1, self.mode)]

        shifting_frame_width = 0.1 # fraction of screen
        if not NP.isnan(NP.array(px, dtype = float)) and NP.any(NP.abs(px - limits[0,:]) < shifting_frame_width*extent[0]):
            self.image_ax.set_xlim([int(px-extent[0]//2),int(px+extent[0]//2)])

        if not NP.isnan(NP.array(py, dtype = float)) and NP.any(NP.abs(py - limits[1,:]) < shifting_frame_width*extent[1]):
            self.image_ax.set_ylim([int(py-extent[1]/2),int(py+extent[1]/2)])


    def UpdatePlot(self, reset = False, hold = False, auto = False):
        ## update the data display axes of the GUI

        # remove previous data
        self.EmptyPlot(reset)

        # auto pan view if tracking click is close to margin
        # comment in this section to enable auto pan (experimental)
        if False and (self.mode > 0) \
             and (self.imagebuffer.pointer > 0) \
             and (not self.playing) \
             and (not reset)\
             :
            self.AutoPanOnMargin()


        # show the current frame, either color or grayscale
        if self.grayscale:
            # grayscale image
            self.image_ax.imshow(self.imagebuffer.GetCurrent(), origin='upper', zorder = 0, cmap = 'gray')
        else:
            # RGB image (untested)
            self.image_ax.imshow(self.imagebuffer.GetCurrent(), origin='upper', zorder = 0) 


        if auto:
            self.image_ax.add_patch(MP.patches.Rectangle((self.pos[0]-self.roi_halfedge, self.pos[1]-self.roi_halfedge) \
                                    , 2*self.roi_halfedge \
                                    , 2*self.roi_halfedge \
                                    , linewidth=1, edgecolor='y', facecolor="none"))


        # plot trace markers of previously tracked points:
        # (A) of the currently selected tracemark 
        rewind = self.playing and self.modifiers['control']
        if not (self.mode == 0):
            if not self.single_image_mode:
                self.PlotPoints(  key = self.mode \
                                , interval = [-5, 0] \
                                , color = (0.7,0.3,0.3) \
                                , alpha = 0.3 if rewind else 1. \
                                )
            if self.imagebuffer.pointer < len(self.imagebuffer)-1:
                self.PlotPoints(  key = self.mode \
                                , interval = [0,5] \
                                , color = (0.7,0.3,0.3) \
                                , alpha = 1.0 if rewind else 0.3\
                                )

            if auto:  
                self.PlotPoints(  key = self.mode \
                                , interval = [self.keyframe - self.imagebuffer.pointer, 0] \
                                , color = (0.3,0.7,0.3) \
                                , alpha = 1.0 \
                                )

        # (B) of the other, unselected tracemarks
        for other_key in [key \
            for key in tracking_modes.keys() \
                if (key != self.mode) \
                and not (key == 0) \
                and not (key in hide_points) \
                and not self.single_image_mode
                ]:
            self.PlotPoints(  key = other_key \
                            , interval = [-5,0] \
                            , color = (0.3,0.7,0.5) \
                            , alpha = 0.3 if rewind else 1. \
                            )
            if self.imagebuffer.pointer < len(self.imagebuffer)-1:
                self.PlotPoints(  key = other_key \
                                , interval = [0,5] \
                                , color = (0.3,0.7,0.5) \
                                , alpha = 1.0 if rewind else 0.3\
                                )

        # if existing: plot epiplines
        if self.current_epilines is not None:
            self.PlotEpilines()

        # progress axis
        if len(self.imagebuffer) > 1:
            self.trackstatus = self.progress_ax.pcolorfast( \
                                  range(len(self.imagebuffer)) \
                                , range(len(tracking_modes)) \
                                , self.data[self.view].isnull().values.T \
                                , cmap = MP.colors.LinearSegmentedColormap("boolpcolor", trackstatus_color) \
                                , vmin=0, vmax=1, zorder = 0)
        if not (self.mode == 0):
            # indicate selected mode
            self.progress_ax.axhline(self.mode - 0.5, ls = '-', lw = 0.5, color = 'k', alpha = 0.3, zorder = 50)

        # time indicator
        self.progress_ax.axvline( x = self.imagebuffer.pointer \
                        , ymin = 0, ymax = len(tracking_modes)-1 \
                        , zorder = 100 \
                        , ls = '-' \
                        , color = (0.6, 0.2, 0.2) \
                        , alpha = 0.5
                        , linewidth = 0.5 \
                        )

        if self.keyframe is not None:
            self.progress_ax.axvline( x = self.keyframe \
                            , ymin = 0, ymax = len(tracking_modes)-1 \
                            , zorder = 100 \
                            , ls = ':' \
                            , color = (0.6, 0.6, 0.6) \
                            , alpha = 0.5
                            , linewidth = 0.5 \
                            )

        if not hold:
            self._canvas.draw()
        # self._canvas.blit(self.image_ax.bbox) # blitting conflicts with the progressive tracking mode
        
        # # mouse cursor ## deactivated to improve performance
        # self.cursor = MPW.Cursor( self.image_ax \
        #                         , useblit = True \
        #                         , color = (0.7,0.6,0.9) \
        #                         , linewidth = 0.2 \
        #                         , linestyle = '-' \
        #                         , zorder = 50 \
        #                         , alpha = 0.5 \
        #                         )

            self.indicator.__setitem__('text', self.GetIndicatorText() ) # update status text

        self.update() # this command enables gui response during progressive tracking.


    def ShowHelp(self, event = None):
        # display the help annotations
        bbox = dict(boxstyle = "Square", fc = "w", ec = "none", lw=0, alpha = 0.6)
        for note in self.help:
            background = NP.mean(self.imagebuffer.GetCurrent().ravel())
            color = '0' if background > 0.5 else '1'
            note.Draw(color, fontsize = 6, bbox = bbox)

        self._canvas.draw()
        self.update()



#______________________________________________________________________________
### callbacks
    def NOOP(self, *arge, **kwargs):
        # do nothing
        pass


    def ChangePointer(self, hold = False, *args, **kwargs):
        ## link to the control of the active frame in the image buffer
        self.imagebuffer.ChangeImgPointer(*args, **kwargs)
        self.UpdatePlot(hold = hold)

    def Forward(self, evt = None):
        ## proceed in the video
        self.ChangePointer(change = +10 if self.modifiers['shift'] else +1)

    def Backward(self, evt = None):
        ## go back in the video
        self.ChangePointer(change = -10 if self.modifiers['shift'] else -1)


    def GoToStart(self, evt = None):
        ## go to first tracked frame or beginning of video 

        # find the first frame of the current marker
        if self.mode == 0:
            first_tracked_current = 0
        else:
            first_tracked_current = NP.argmin(NP.all(PD.isnull( \
                                            self.data[self.view].loc[:, [f"{tracking_modes[self.mode]}_{coord}" for coord in coordinates]].values \
                                        ), axis = 1))

        # find the first frame with data at all
        first_tracked_all = NP.argmin(NP.all(PD.isnull(self.data[self.view].values), axis = 1))

        if (first_tracked_current >= first_tracked_all) \
            and ((first_tracked_current is None) or (first_tracked_current < self.imagebuffer.pointer)):
            # go to that frame...
            self.imagebuffer.GoToFrame(first_tracked_current)

        elif (first_tracked_all is None) or (first_tracked_all < self.imagebuffer.pointer):
            # go to that frame...
            self.imagebuffer.GoToFrame(first_tracked_all)

        else:
            # ... or go to the start.
            self.imagebuffer.GoToFrame(0)

        self.UpdatePlot()


    def SetMode(self, mode):
        ## change tracked marker = mode
        self.mode = mode

        if self.EpilineCalculator is not None:
            self.UpdateEpilines()

        self.UpdatePlot()


    def DropdownModeSelect(self, *_):
        ## change the tracking mode by a dropdown
        new_mode = self.modelabels.index(self.dropdownmode.get())
        self.SetMode(new_mode)

    def HideModeOptionsMenu(self, _):
        # hide tracking mode selection if the menu had previously been popped up
        if self.mode_optionsmenu['menu'].winfo_viewable():
            self.mode_optionsmenu['menu'].unpost() # hide menu
            self.mode_optionsmenu['menu'].grab_release() # try to release the focus

            self.mode_optionsmenu.focus_set() # try to release the focus
            self.mode_optionsmenu.grab_set_global() # try to release the focus

            self.update()


    def UpdateEpilines(self):
        # updates the epipolar lines 
        # for the current mode
        # projected to the other cameras
        

        # no tracking
        if self.mode == 0:
            return

        # loop other cameras
        for crosscam in self.cameras:
            if self.view == crosscam:
                continue
                
            # get fundamental matrix
            funmat = self.fundamental_matrices[(self.view, crosscam)]
            
            # get points
            points = self.data[crosscam].loc[:, [f"{tracking_modes[self.mode]}_{coord}" for coord in coordinates]]

            # add a lines for cross image
            lines = self.EpilineCalculator(points.values.astype(NP.float32).reshape(-1,1,2), 2, funmat).reshape(-1,3)

            for frame_nr, line in enumerate(lines):
                self.current_epilines[(self.view, crosscam)].loc[frame_nr, ['y0', 'y1']] = NP.array([ \
                                          -line[2]/line[1] \
                                        , -(line[2]+line[0]*self.imagebuffer.shape[2])/line[1] \
                                        ])


    def SwitchView(self, evt = None):
        # change the camera if multiple perspectives were loaded

        # store old view
        old_view = self.view
        
        # adjust the view pointer
        self.view = self.cameras[(self.cameras.index(self.view) + (-1 if self.modifiers['shift'] else 1)) % (len(self.cameras))]
        
        # update the link to one of the buffers
        self.imagebuffer = self.all_buffers[self.view]
        self.imagebuffer.GoToFrame(self.all_buffers[old_view].pointer)

        # update epipolar lines
        if self.EpilineCalculator is not None:
            self.UpdateEpilines()

        # done!
        self.UpdatePlot()



    def SwitchMode(self, event):
        ## change the mode to previous/next

        # get current mode
        current = self.modelist.index(self.mode)

        # next or previous
        new = current
        if event.key == 'up':
            new += 1
        elif event.key == 'down':
            new -= 1
        else:
            return

        # set new mode
        self.SetMode(self.modelist[new % len(self.modelist)])


    def Zoom(self, evt):
        ## zooming with keyboard shortcuts, round mouse cursor position
        if self.playing:
            return

        # zoom = change of extent if 'i' or 'p' are pressed
        dz = 0.1 if evt.key == 'i' else (-0.1 if evt.key == 'p' else 0.0)

        # current extent
        x_curr = self.image_ax.get_xlim()
        y_curr = self.image_ax.get_ylim()

        # position
        if self.pos is None:
            c = [NP.mean(x_curr), NP.mean(y_curr)]
        else:
            c = self.pos

        # scale
        extent = NP.array([x_curr[1]-x_curr[0], y_curr[1]-y_curr[0]])
        extent *= (1+dz)

        # don't scale out
        extent[0] = NP.min([extent[0], self.imagebuffer.shape[1]])
        extent[1] = NP.min([extent[1], self.imagebuffer.shape[2]])
        
        # prepare new basis
        try:
            new_x = NP.min([NP.max([0,c[0] - extent[0]/2]), self.imagebuffer.shape[1]-extent[0] ])
            new_y = NP.min([NP.max([0,c[1] - extent[1]/2]), self.imagebuffer.shape[2]-extent[1] ])

            # adjust limits
            self.image_ax.set_xlim([new_x, new_x+extent[0] ])
            self.image_ax.set_ylim([new_y, new_y+extent[1] ])
        except TypeError as te:
            print (te)
            print (c, extent, self.imagebuffer.shape)

        # update
        self.UpdatePlot()


    def PanZoomShortcut(self, event):
        # toggle pan/zoom mode
        if (not self.playing) and (not (event.inaxes == self.progress_ax)) and (not self.clicking):
            if not self.toolbar.mode == 'pan/zoom':
                self.toolbar.pan()
            elif self.toolbar.mode == 'pan/zoom':
                self.toolbar.pan()


    def AxisEnter(self, event):
        ## release pan/zoom tools if progress axis is entered
        # print (dir(event))
        if event.inaxes == self.progress_ax:
            # print (self.toolbar.mode)
            if self.toolbar.mode == 'pan/zoom':
                self.toolbar.pan()
            if self.toolbar.mode == 'zoom rect':
                self.toolbar.zoom()
            # print ('released')


    def PanLeft(self, _):
        self.PanLeftRight(rightwards = False)

    def PanRight(self, _):
        self.PanLeftRight(rightwards = True)

    def PanLeftRight(self, rightwards):
        x_step = 10
        if self.modifiers['shift']:
            x_step *= 10

        if rightwards: 
            x_step *= -1

        if self.modifiers['control']:
            LimFcns = [self.image_ax.set_ylim, self.image_ax.get_ylim]
        else:
            LimFcns = [self.image_ax.set_xlim, self.image_ax.get_xlim]
        LimFcns[0](int(x_step) + NP.array(LimFcns[1](), dtype = int))

        self.UpdatePlot()


#______________________________________________________________________________
### Progressive Tracking
# letting time flow (i.e. video playback), while following a point with the mouse. 

    def TogglePlay(self, evt = None):
        ## start/stop progressive tracking
        # print ("toggle play")
        
        if self.auto_tracking:
            self.auto_tracking = False
            return

        self.playing = not (self.playing)

        # storing the coordinates 
        if self.playing:
            # print ("started")
            self.ProgressiveTracking()
        else:
            # print ("stopped")
            pass # (only used for debugging)


    def Debug(self):
        # show output relevant for debugging
        print (self.imagebuffer.pointer \
                , self.imagebuffer.files[self.imagebuffer.pointer] \
                , self.pos \
                , TI.time() \
                )

    def ProgressiveTracking(self, evt = None):
        ## progressive tracking, i.e. playing the video and capturing mouse pointer.
        while self.playing:
            # wait for a time that is equal to "speed"
            TI.sleep(self.speed)
            # print (evt, '{}, {}'.format(*self.pos), self.playing)
            
            # stop at the end
            if self.imagebuffer.pointer >= len(self.imagebuffer)-1:
                self.playing = False
                return

            # self.Debug()
            # progress
            direction = -1 if self.modifiers['control'] else +1
            step = +10*direction if self.modifiers['shift'] else +1*direction
            self.ChangePointer(hold = True, change = step)
                        
            # store mouse position
            if not (self.mode == 0):
                self.update() # to make sure that the mouse has not moved during video update

                xy = self.pos
                if self.blobdet.get():
                    img = self.imagebuffer.GetCurrent()
                    if not self.grayscale:
                        img = ColorToGray(img)

                    xy, r = BlobDetect(SKIU.invert(img), center = xy, frame = self.roi_halfedge \
                                       , **blobdict)

                    c = MPP.Circle((xy[0], xy[1]), r, color='y', linewidth=0.5, fill=False)
                    self.image_ax.add_patch(c)

                self[(self.imagebuffer.pointer-step, self.mode)] = xy

            # show the new image immediately after the position was stored. 
            self._canvas.draw()
            self.indicator.__setitem__('text', self.GetIndicatorText() ) # update status text
            # self.imagebuffer.Interrupt()


    def GetCoordinates(self, evt):
        ## callback that handles coordinate storage in progressive tracking
        if evt is None:
            return
        self.pos = (evt.xdata, evt.ydata)
        # print('{}, {}'.format(*self.pos))



    def ChangeSpeed(self, evt = None):
        ## adjust speed in progressive tracking
        # note that speed is stored reciprocally: 
        #   it is taken as log of a number smaller than one.
        #   speed is equal to the time interval between two frames in playback mode

        n_steps = 10
        step = 1.0/n_steps # steps in which speed is adjusted
        if evt is not None:
            if evt.name == 'scroll_event':
                command = evt.button
            else:
                # "key_press_event"
                command = evt.key

            # inverted because of log
            speedchange = { \
                          '+': -step
                        , '-': +step
                        , 'up': -step
                        , 'down': +step
                        }.get(command, None)
            
            if speedchange is None:
                # should not happen
                return
        else:
            return

        if self.modifiers['shift']:
            speedchange *= 10

        # cap max/min speed
        self.speed = NP.max([10**(-max_speed), NP.min([1, 10**(speedchange+NP.log10(self.speed))])])

        # adjust status indicator
        self.indicator.__setitem__('text', self.GetIndicatorText() )



    def DeleteData(self):
        ## delete data vfrom the current tracemark

        # cancel if "no tracking"
        if self.mode == 0:
            self.StdOut('nothing to delete! (please select a tracking mode with data)')
            return

        to_delete = [TK.IntVar(), TK.IntVar()]
        to_delete[0].set(0)
        to_delete[1].set(len(self.all_buffers[self.view]))

        # get the user input
        popup = DeleteDataPopup(self, tracking_modes[self.mode], *to_delete)
        self.wait_window(popup.window)

        # extract the deletion interval
        delete_interval = [NP.min([len(self.imagebuffer),NP.max([0,t_del.get()])]) for t_del in to_delete]

        # check if it is valid
        if NP.diff(delete_interval) <= 0:
            self.StdOut('nothing to delete!')
            return

        # delete the data
        self.data[self.view].loc[delete_interval[0]:delete_interval[1], [f"{tracking_modes[self.mode]}_{coord}" for coord in coordinates]] = NP.nan
        self.StdOut('deleted [{}, {}] of {}!'.format(*(delete_interval + [tracking_modes[self.mode]])))

        # update display
        self.UpdatePlot()



    def ToggleNoClick(self):
        ## disable tracking by clicking
        # https://stackoverflow.com/questions/50276202/tkinter-checkbutton-not-working
        self.noclick.set(not self.noclick.get())
        self.noclick_cb.toggle()


    def ToggleShowEpilines(self, evt = None):
        ## disable tracking by clicking
        # https://stackoverflow.com/questions/50276202/tkinter-checkbutton-not-working
        self.show_epilines.set(not self.show_epilines.get())
        if evt is None:
            # for some reason, cb command does not toggle by itself.
            self.show_epilines_cb.toggle()

        self.UpdatePlot()


    def ToggleBlobDetection(self, evt = None):
        ## disable tracking by clicking
        # https://stackoverflow.com/questions/50276202/tkinter-checkbutton-not-working
        self.blobdet.set(not self.blobdet.get())
        if evt is None:
            # for some reason, cb command does not toggle by itself.
            self.blobdet_cb.toggle()

        self.update()



#______________________________________________________________________________
### Auto Tracking

    def AutoTrack(self, evt = None):

        if self.auto_tracking:
            # stop procedure
            self.auto_tracking = False
            return

        if (self.mode == 0) or (self.playing):
            return

        self.InsertKeyframe()
        self.UpdatePlot(auto = True)

        self.StdOut('autotracking...')

        px, py = map(int, self.pos)
        self[(self.imagebuffer.pointer, self.mode)] = NP.array([px, py])
        rhel = int(self.roi_halfedge)

        start_rect = DL.rectangle(px - rhel, py - rhel, px + rhel, py + rhel)
        start_crop = NP.array([px - rhel, py - rhel, px + rhel, py + rhel])


        start_frame = self.imagebuffer.GetCurrent()
        start_roi = ExtractRegionOfInterest(start_frame, start_crop)

        start_center, r0 = BlobDetect(SKIU.invert(start_frame), center = (px, py), frame = self.roi_halfedge \
                                    , normalize = False \
                                    , **blobdict \
                                   )

        if not self.grayscale:
            start_roi = ColorToGray(start_roi)
        start_roi = ImageTo8Bit(start_roi)

        
        # TODO: plot rectangle


        self.auto_tracking = True
        car = DL.correlation_tracker()
        car.start_track(ImageTo8Bit(start_frame), start_rect)

        orb = SKIF.ORB(n_keypoints = self.n_features)
        exit_cause = ''
        while self.auto_tracking:

            if (self.imagebuffer.pointer - self.keyframe) % 10 == 0:
                self.StdOut(f"autotracked {self.imagebuffer.pointer - self.keyframe} frames")
                self.update()

            self.imagebuffer.ChangeImgPointer(change = +1)

            img = self.imagebuffer.GetCurrent()

            car.update(ImageTo8Bit(img))
            autopos = RectToPosition(car.get_position())


            roi = ExtractRegionOfInterest(img, autopos)
            if not self.grayscale:
                roi = ColorToGray(roi)

            if self.blobdet.get():
                # xy = NP.array([(autopos[2]+autopos[0])/2, (autopos[3]+autopos[1])/2])
                xy = roi.shape[1]/2, roi.shape[0]/2
                xy, r = BlobDetect(SKIU.invert(roi), center = (xy[0], xy[1]), r0 = r0, frame = None#self.roi_halfedge \
                                    , normalize = False \
                                   , **blobdict)

                xy = (xy[0]+autopos[0], xy[1]+autopos[1])

                if r == 0:
                    self.auto_tracking = False
                    exit_cause = 'no blob found'
                    break

            else:
                xy = NP.array([(autopos[2]+autopos[0])/2, (autopos[3]+autopos[1])/2])

            # set data to the center of the square
            self[(self.imagebuffer.pointer-1, self.mode)] = xy


            start1, start2 = MatchedFeatures(orb, start_roi, ImageTo8Bit(roi))

            # print ('#'*10)
            # print ('\t', self.imagebuffer.pointer)

            if self.imagebuffer.pointer == len(self.imagebuffer)-1:
                # end of video
                self.auto_tracking = False
                exit_cause = 'end of video.'
                break

            # print ('\t', len(start1))

            if len(start1) <= 2:
                # too few matches
                self.auto_tracking = False
                exit_cause = f'too few feature matches ({len(start1)}).'
                break
            
            # print ('\t', ProcrustesDistance(start2, start1))

            pd = ProcrustesDistance(start2, start1)
            if pd > self.pd_threshold:
                self.auto_tracking = False
                exit_cause = f'too low feature similarity ({len(start1)} matches, pd = {pd:.2f}).'
                break



        self.UpdatePlot(auto = True)
        self.StdOut(f'autotracking stopped, {exit_cause}')
        self.update()



    def AdjustAutoParameters(self, evt = None):
        # ROI control

        # adjsut the pixel square used for correlation autotracking
        if evt.key == '[':
            self.roi_halfedge -= 8
            self.roi_halfedge = NP.max([8, self.roi_halfedge]) 

        elif evt.key == ']':
            self.roi_halfedge += 8
            self.roi_halfedge = NP.min([self.roi_halfedge, self.imagebuffer.shape[1]//2, self.imagebuffer.shape[2]//2])

        # adjust feature count
        if evt.key == '<':
            self.n_features -= 8
            self.n_features = NP.max([8, self.n_features]) 

        elif evt.key == '>':
            self.n_features += 8

        if evt.key == '{':
            self.pd_threshold -= 0.08
            self.pd_threshold = NP.max([0.02, self.pd_threshold]) 

        elif evt.key == '}':
            self.pd_threshold += 0.08
            self.pd_threshold = NP.min([1.92, self.pd_threshold]) 


        self.StdOut(f'<-> {self.roi_halfedge}; n = {self.n_features}, pd < {self.pd_threshold:.2f}')
        self.update()


    def InsertKeyframe(self, evt = None):
        self.keyframe = self.imagebuffer.pointer

    def ReturnToKeyframe(self, evt = None):
        new_keyframe = self.imagebuffer.pointer
        self.imagebuffer.GoToFrame(self.keyframe)
        self.keyframe = new_keyframe
        self.UpdatePlot()
        


#______________________________________________________________________________
### data I/O
    def Save(self, evt = None):
        ## saving data
        if self.data[self.view] is None:
            self.StdOut("nothing to save.")
            return

        # query the file to store
        if self.modifiers['shift'] or (self.outfile is None):
            self.outfile = TKF.asksaveasfilename( \
                                  defaultextension = ".csv" \
                                , filetypes = [('comma separated values', '.csv'), ('all files', '.*')] \
                                , title = 'Select a file to save the traces.' \
                                , initialdir = '.' \
                                , initialfile = f"{save_prefix}_{self.folder[0].split('/')[-1]}_{label}.csv" \
                                )

            # break if user did not select a file
            if len(self.outfile) == 0:
                self.StdOut('not saved!')
                print('not saved!')
                self.outfile = None
                return
    
        # self.data[self.view].drop(columns = ['frame_nr'], inplace = True)
        index_column = 'point_nr' if self.single_image_mode else 'frame_nr'

        for view in self.data.keys():
            if 'frame_nr' in self.data[view].columns:
                self.data[view].set_index('frame_nr', inplace = True, drop = True)
            
            self.data[view].index.name = index_column
        
        # assemble data to store
        storedata = []
        if len(self.data) == 1:
            storedata = self.data[self.view].copy().reset_index(drop = False, inplace = False)
            storedata['folder'] = self.folder[self.view]
            storedata.set_index(['folder', index_column], inplace = True)
        else:
            for view, data_frame in self.data.items():
                data = data_frame.copy().reset_index(drop = False, inplace = False)
                data['folder'] = self.folder[view]
                data.set_index(['folder', index_column], inplace = True)
                storedata.append(data)

            storedata = PD.concat(storedata)


        # store data to file
        try:
            storedata.to_csv( \
                                  self.outfile \
                                , sep = ';' \
                                , header = True \
                                , float_format = '%.6f' \
                                , index = True \
                                )
            self.StdOut("saved! %s" % (self.outfile))
            print(f"saved! {self.outfile}")
            self.saved = True 
        except PermissionError as perr:
            self.StdOut('Not saved (permission error)!')
            print(perr)


    def Load(self, evt = None):
        ## loading previously tracked data
        infile = TKF.askopenfilename( \
                                  defaultextension = ".csv" \
                                , filetypes = [('track data', '.csv'), ('all files', '.*')] \
                                , title = 'Load previously tracked tracc.' \
                                )

        # selection was unsuccessful
        if (infile is None) or (infile == '') or (len(infile) == 0): # asksaveasfile return `None` if dialog closed with "cancel".
            return

        # get the data from loaded file
        trace_df = PD.read_csv( infile, sep = ';' )#.set_index('frame_nr', inplace = False, drop = True)
        
        # check if rows match
        if trace_df.shape[0] != self.imagebuffer.shape[0] * len(self.all_buffers):
            print ("video does not match!")
            return

        # confirm overwrite
        if not TKM.askyesno("load", "really overwrie data?"):
            return


        # different index column for single image mode
        index_column = 'point_nr' if self.single_image_mode else 'frame_nr'

        # get data per camera/view
        if 'folder' in trace_df.columns:
            loaded_traces = {}
            for fld in self.folder:
                df = trace_df.loc[trace_df['folder'].values == fld, :].copy()
                df.drop(columns = ['folder'], inplace = True)
                
                if index_column in df.columns:
                    df.set_index(index_column, inplace = True)
                    df = df.loc[self.data[0].index, :]
                else:
                    df.index = self.data[0].index

                if ('point_nr' in df.columns) or ('frame_nr' in df.columns):
                    df.drop(columns = ['point_nr', 'frame_nr'], inplace = True)
                loaded_traces[self.folder.index(fld)] = df

        else:
            loaded_traces = {0: trace_df}


        for view in self.cameras:
            # only load tracemark that is in tracking modes
            self.data[view] = PD.DataFrame( index = loaded_traces[view].index.values \
                                    , columns = \
                                        [f'{tracking_modes[tm]}_{coord}'  \
                                         for tm in sorted(tracking_modes.keys()) for coord in coordinates if tm > 0 \
                                        ] )
            self.data[view].index.name = 'frame_nr'

            for col in loaded_traces[view].columns:
                tm = "_".join(col.split('_')[:-1])
                if tm in tracking_modes.values():
                    self.data[view].loc[:, col] = loaded_traces[view][col].values

            # print (self.data[view])

        # update epilines
        if self.EpilineCalculator is not None:
            self.UpdateEpilines()

        # update display
        self.UpdatePlot()


#______________________________________________________________________________
### gui creation
    def MakeGUI(self):
        ## prepare the GUI

        # http://matplotlib.org/users/event_handling.html
        self.keyboard_shortcuts = { }
        self.keyboard_shortcuts[' '] = self.TogglePlay
        # self.bind('<space>', self.TogglePlay)
        self.keyboard_shortcuts['right'] = self.Forward
        self.keyboard_shortcuts['left'] = self.Backward
        self.keyboard_shortcuts['up'] = self.SwitchMode
        self.keyboard_shortcuts['down'] = self.SwitchMode
        self.keyboard_shortcuts['home'] = self.GoToStart
        self.keyboard_shortcuts['v'] = self.SwitchView
        # self.keyboard_shortcuts['tab'] = self.SwitchView # does not work, Tab is bound by tk
        self.keyboard_shortcuts['e'] = self.ToggleShowEpilines
        self.keyboard_shortcuts['b'] = self.ToggleBlobDetection
        self.keyboard_shortcuts['s'] = self.Save
        self.keyboard_shortcuts['l'] = self.Load
        self.keyboard_shortcuts['+'] = self.ChangeSpeed
        self.keyboard_shortcuts['-'] = self.ChangeSpeed
        self.keyboard_shortcuts['j'] = self.PanLeft
        self.keyboard_shortcuts['k'] = self.PanZoomShortcut
        self.keyboard_shortcuts['l'] = self.PanRight
        self.keyboard_shortcuts['i'] = self.Zoom
        self.keyboard_shortcuts['o'] = self.Zoom
        self.keyboard_shortcuts['p'] = self.Zoom
        self.keyboard_shortcuts['a'] = self.AutoTrack
        self.keyboard_shortcuts['<'] = self.AdjustAutoParameters
        self.keyboard_shortcuts['>'] = self.AdjustAutoParameters
        self.keyboard_shortcuts['['] = self.AdjustAutoParameters
        self.keyboard_shortcuts[']'] = self.AdjustAutoParameters
        self.keyboard_shortcuts['{'] = self.AdjustAutoParameters
        self.keyboard_shortcuts['}'] = self.AdjustAutoParameters
        self.keyboard_shortcuts['.'] = self.InsertKeyframe
        self.keyboard_shortcuts['pageup'] = self.ReturnToKeyframe
        self.keyboard_shortcuts['q'] = self.Quit
        self.keyboard_shortcuts['f1'] = self.ShowHelp
        self.keyboard_shortcuts['escape'] = self.Quit

        ### remove overlapping shortcut keys from matplotlib toolbar
        # print ([k for  ])
        for command in [cmd for cmd in MPP.rcParams.keys() if 'keymap' in cmd]:
            for shortkey in MPP.rcParams[command]:
                if shortkey in self.keyboard_shortcuts:
                    MPP.rcParams[command].remove(shortkey)
                    # print (command, shortkey)

        # print ("\n".join([f'{key}: {MPP.rcParams[key]}' for key in ['keymap.all_axes', 'keymap.back', 'keymap.copy' \
        #         , 'keymap.forward', 'keymap.fullscreen', 'keymap.grid' \
        #         , 'keymap.grid_minor', 'keymap.help', 'keymap.home' \
        #         , 'keymap.pan', 'keymap.quit', 'keymap.quit_all', 'keymap.save' \
        #         , 'keymap.xscale', 'keymap.yscale', 'keymap.zoom']]))

        # alternative binding:
        # self.bind('q', self.Quit)


        ### canvas
        self._canvas = FigureCanvasTkAgg(self.fig, master = self)
        self._canvas.mpl_connect('motion_notify_event', self.GetCoordinates)
        self._canvas.mpl_connect('key_press_event', self.PressKey)
        self._canvas.mpl_connect('key_release_event', self.ReleaseKey)

        self._canvas.mpl_connect('button_press_event', self.ClickCapture)
        self._canvas.mpl_connect('button_release_event', self.ClickRelease)
        self._canvas.mpl_connect('scroll_event', self.MousewheelCapture)

        self._canvas.mpl_connect('axes_enter_event', self.AxisEnter)
        # help(self._canvas.mpl_connect) # axes_enter_event, axes_leave_event


        ### toolbar, taken from matplotlib
        self.toolbar = NavigationToolbar2Tk( self._canvas, self )
        # self.toolbar.mpl_connect('key_press_event', self.PressKey)
        # self.toolbar.mpl_connect('key_release_event', self.ReleaseKey)

        ### menu
        self.button_height = 1
        self.menu = TK.Frame(master = self.toolbar) # self
        self.menu.pack(side = TK.LEFT, fill = TK.X, expand = False)
        
        # add buttons
        TK.Button(master = self.menu, text = 'quit', command = self.Quit \
                                , height = self.button_height, padx = 20 \
                                ).pack(side = TK.RIGHT) 
 
        TK.Button(master = self.menu, text = 'load', command = self.Load \
                                , height = self.button_height, padx = 10 \
                                ).pack(side = TK.RIGHT) 
        TK.Button(master = self.menu, text = 'save', command = self.Save \
                                , height = self.button_height, padx = 10 \
                                ).pack(side = TK.RIGHT) 

        TK.Button(master = self.menu, text = 'del', command = self.DeleteData \
                                , height = self.button_height, padx = 10 \
                                ).pack(side = TK.RIGHT) 


        # a variable to trace the dropdown mode selection
        self.dropdownmode = TK.StringVar()
        self.dropdownmode.set(tracking_modes[0])
        self.dropdownmode.trace("w", self.DropdownModeSelect)


        self.mode_optionsmenu = TK.OptionMenu( \
                                  self.menu \
                                , self.dropdownmode \
                                , *self.modelabels \
                                )
        self.mode_optionsmenu.pack(side = TK.RIGHT) 
        # help(self.mode_optionsmenu.bind)
        self.mode_optionsmenu.bind('<ButtonPress>', self.HideModeOptionsMenu)

        # disable click tracking
        self.noclick = TK.BooleanVar()
        self.noclick.set(False)
        self.noclick_cb = TK.Checkbutton(master = self.menu, text = 'noclick' \
                                , variable = self.noclick \
                                , command = self.ToggleNoClick \
                                , height = self.button_height, padx = 10 \
                                )
        self.noclick_cb.pack(side = TK.RIGHT) #,

        self.show_epilines = TK.BooleanVar()
        self.show_epilines.set(False)
        if self.EpilineCalculator is not None:
            self.show_epilines_cb = TK.Checkbutton(master = self.menu, text = 'epilines' \
                                    , variable = self.show_epilines \
                                    , command = self.ToggleShowEpilines \
                                    , height = self.button_height, padx = 10 \
                                    )
            self.show_epilines_cb.pack(side = TK.RIGHT) #,

        self.blobdet = TK.BooleanVar()
        self.blobdet.set(False)
        self.blobdet_cb = TK.Checkbutton(master = self.menu, text = 'blobdet' \
                                , variable = self.blobdet \
                                , command = self.ToggleBlobDetection \
                                , height = self.button_height, padx = 10 \
                                )
        self.blobdet_cb.pack(side = TK.RIGHT) #,


        # time control
        TK.Button(master = self.menu, text = '>>', command = self.Forward \
                                , height = self.button_height, padx = 10 \
                                ).pack(side = TK.RIGHT) 

        TK.Button(master = self.menu, text = 'play', command = self.TogglePlay \
                                , height = self.button_height, padx = 10 \
                                ).pack(side = TK.RIGHT) 

        TK.Button(master = self.menu, text = '<<', command = self.Backward \
                                , height = self.button_height, padx = 10 \
                                ).pack(side = TK.RIGHT) 



        # add indicator 
        self.indicator = TK.Label( master = self.toolbar, text = 'starting...' ) 
        self.indicator.pack( side = TK.LEFT )
        self.indicator.__setitem__('text', self.GetIndicatorText() )

        # pack canvas and toolbar
        self.toolbar.pack(side = TK.TOP, fill = TK.X, expand = False) 
        self.toolbar.update()
        
        self._canvas.get_tk_widget().pack(side = TK.TOP, fill = TK.BOTH, expand = True)
        self._canvas.draw()

        # make help annotations
        self.PrepareHelpTexts()



    def PrepareHelpTexts(self):
        # bring in annotations for the help texts. 

        helptexts = { \
              'System': None \
            , '[escape]|[q]': 'quit' \
            , '[F1]': 'show/hide help' \
            , '[s]': 'save (to previously selected file)' \
            , '[shift]+[s]': 'save as' \
            , '[l]': 'load' \
            , '[down]/[up]': 'previous/next tracking mode' \
            , '[1-9]': 'tracking modes quick select' \
            , '[0]': 'playback only ("no tracking" mode)' \
            , '[v]': 'switch view (if in multi camera mode)' \
            , '[click]': 'on image: track; on time bar: jump to frame' \
            , '\separator1': None \
            , 'Time': None \
            , '[space]': 'play/pause (progressive tracking)\n[ctrl] while playing: play backwards!' \
            , '[left]/[right]': 'previous/next frame ([shift] jump 10 frames)' \
            , '[home]': 'jump to start' \
            , '[+]/[-]': 'accelerate/decelerate playback; [shift] increases steps' \
            , 'mouse wheel': 'adjust playback speed, even while playing' \
            , '\separator2': None \
            , 'Image': None \
            , '[i]/[o]/[p]': 'zoom out/center view/zoom in' \
            , '[k]': '(de-)activate pan/zoom tool' \
            , '[j]/[l]': 'shift view left/right ([shift] increase pixel step)\n[ctrl] shift view up/down' \
            , '[a]': 'attempt auto digitization' \
            , '[[]/[]]': 'adjust auto ROI size' \
            , '[<]/[>]': 'adjust auto ROI feature count' \
            , '[{]/[}]': 'adjust feature procrustes threshold' \
            }

        extra_linebreaks = 1 # for the '\n' in one of the annotations
        y_step = 1./(len(helptexts) + 1 + extra_linebreaks)

        current_y = -y_step/2 # start half a step above screen
        for key, txt in helptexts.items():
            current_y += y_step

            if (key[:10] == '\separator'):
                continue

            self.help.append(HelpAnnotation(self.image_ax, (0.05, 1. - current_y), key, color = 'k'))

            if (txt is None):
                continue
            self.help.append(HelpAnnotation(self.image_ax, (0.28, 1. - current_y), txt, color = 'k'))

            # print (current_y, txt)
            if '\n' in txt:
                current_y += y_step


#______________________________________________________________________________
### Epipolar Geometry
    def EpipolarGeometry(self, image_point_file):
        # establish epipolar geometry
        if image_point_file is not None:
            self.fundamental_matrices, self.EpilineCalculator = PrepareEpipolarGeometry(image_point_file)
            # calculate epilines from image points
            self.current_epilines = {key: PD.DataFrame(columns = [ f"{coord}{nr}" \
                                                  for coord in ['y'] \
                                                  for nr in [0,1] \
                                              ] \
                            , index = NP.arange( self.imagebuffer.shape[0] )) \
                            for key in self.fundamental_matrices.keys()}

        else:
            # no EPGeom
            self.fundamental_matrices = None
            self.EpilineCalculator = None
            self.current_epilines = None




#______________________________________________________________________________
### plotting
    def MakeFigure(self):
        ## prepare the data indicator axes

        # define figure
        self.fig = MPP.figure( \
                                  facecolor = None \
                                # , figsize = (figwidth/2.54*1.25, figheight/2.54*1.25) \
                                , dpi = 300 \
                                )
        # MPP.ion() # "interactive mode". Might be useful here, but i don't know. Try to turn it off later.

        # define axis spacing
        self.fig.subplots_adjust( \
                                  top = 0.99 \
                                , right = 0.99 \
                                , bottom = 0.05 \
                                , left = 0.01 \
                                , wspace = 0.05 \
                                , hspace = 0.05 \
                                )

        # gridspec: rows and columns of each subplot
        gs = self.fig.add_gridspec( \
                          ncols = 1 \
                        , nrows = 2 \
                        , width_ratios = [1] \
                        , height_ratios = [15,1] \
                        )

        # define image and progress axes
        self.image_ax = self.fig.add_subplot(gs[0])
        self.progress_ax = self.fig.add_subplot(gs[1])

        # cosmetics
        FullDespine(self.image_ax)
        FullDespine(self.progress_ax, xlab = True)


#______________________________________________________________________________
### control
    def StdOut(self, text):
        ## decides where status info is printed.
        try:
            self.indicator.__setitem__('text', self.GetIndicatorText(text) ) # status info goes to the indicator.
        except RuntimeError:
            print(text) # status info goes to console instead


    def Quit(self, evt = None):
        ## finishing the tracker
        self.playing = False

        if self.exiting:
            # there has been a previous exit attempt
            return

        # user prompt
        if TKM.askyesno("please don't go...",'really quit?%s' % ('' if self.saved else '\n UNSAVED DATA!')):
            self.exiting = True # confirm exit
            self.imagebuffer.Interrupt() # stop buffering
            MPP.close(self.fig) # close the figure
            SYS.exit() # exit the GUI

        self.exiting = False # user has decided not to exit



################################################################################
### Delete Data                                                              ###
################################################################################
class DeleteDataPopup():
    # a popup to select a range of data that can be deleted
    # https://blog.furas.pl/python-tkinter-how-to-create-popup-window-or-messagebox-gb.html

    def __init__(self, master, tracemark = None, frame_start = None, frame_end = None):

        # assess correct variable preparation
        if (frame_start is None) or (frame_end is None):
            raise IOError('requires tkinter IntVar for interval storage.')

        # create popup window
        self.window = TK.Toplevel(master)
        self.window.title("Delete Frames in %s tracemark" % ('current' if tracemark is None else tracemark))
        
        # call the "quit" function when the window is closed
        self.window.protocol("WM_DELETE_WINDOW", self.Quit)
        self.window.bind('q', self.Quit)
        self.window.bind('<Escape>', self.Quit)

        # define fields and assign variables
        self.fields = ['start (>=)', 'end (<)']
        self.vars = {'start (>=)': frame_start, 'end (<)': frame_end}
        # print (self.vars)

        # validation command
        vcmd = (self.window.register(self.Validate),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')


        # stack the entry boxes
        self.entries = {}
        fields = TK.Frame(self.window)
        for field in self.fields:
            col = TK.Frame(fields)
            lab = TK.Label(col, height=5, text=field, anchor='w')
            self.entries[field] = TK.Entry(col, validate = 'key', validatecommand = vcmd, textvariable = self.vars[field], justify = TK.CENTER)
            col.pack(side=TK.LEFT, expand=TK.NO, padx=50, pady=15) # , fill=TK.X
            lab.pack(side=TK.TOP)
            self.entries[field].pack(side=TK.BOTTOM, expand=TK.YES, fill=TK.X)
            # self.entries[field].insert(TK.END, str(self.vars[field].get()))

        fields.pack(side=TK.TOP, expand=TK.NO)

        # control buttons
        buttons = TK.Frame(self.window)
        button_enter = TK.Button(buttons, text="Delete!", command=self.Enter)
        button_enter.pack(side=TK.LEFT)
        button_close = TK.Button(buttons, text="Cancel", command=self.Quit)
        button_close.pack(side=TK.LEFT)

        buttons.pack(side=TK.TOP, expand=TK.NO) # , fill=TK.X


    def Enter(self, evt = None):
        # good exit path: a range has been entered
        for field in self.fields:
            self.vars[field].set(int(self.entries[field].get()))
        self.window.destroy()

    def Quit(self, evt = None):
        # default exit path: entries are discarded
        for field in self.fields:
            self.vars[field].set(0)
        self.window.destroy()

    def Validate(self, action, index, value_if_allowed,
                       prior_value, text, validation_type, trigger_type, widget_name):
        # validation; see https://stackoverflow.com/questions/4140437/interactively-validating-entry-widget-content-in-tkinter
        if value_if_allowed:
            try:
                float(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False



"""
#######################################################################
### Mission Control                                                 ###
#######################################################################
"""
if __name__ == "__main__":
    ## this is executed when the script is run directly.

    parameters = dict(folder = None) # will open folder selection; you can also just run "Tracker()"
    # parameters = dict(folder = 'testing/fftest') # will open a single data folder
    # parameters = dict(folder = ['testing/cam1', 'testing/cam2', 'testing/cam3']) # will open multiple cameras
    # parameters = dict(folder = ['testing/cal1', 'testing/cal2', 'testing/cal3'], n_points = 128) # folders that contain only a single image
    # parameters = dict(folder = ['testing/beads1', 'testing/beads2'], image_point_file = 'testing/ff_xcams_imagepoints.csv') # multiple cameras and epilines; folders must be in same order as files in "imagepoints"

    # start a tracker GUI with a given folder.
    Tracker(**parameters)


