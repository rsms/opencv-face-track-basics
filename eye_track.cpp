#include <OpenCV/OpenCV.h>
#include <cassert>
#include <iostream>

const char *g_window_name = "Eyes tracker";
const CFIndex g_cascade_name_len = 2048;
char g_cascade_name[g_cascade_name_len] = "abc.xml";

using namespace std;

int main (int argc, char * const argv[]) {
	const int scale = 2,
	          refreshDelay = 10;
	
	// locate haar cascade from inside application bundle
	CFBundleRef mainBundle = CFBundleGetMainBundle();
	assert(mainBundle);
	CFURLRef cascade_url = CFBundleCopyResourceURL(mainBundle, CFSTR("haarcascade_eyes"), CFSTR("xml"), NULL);
	assert(cascade_url);
	if (!CFURLGetFileSystemRepresentation (cascade_url, true, reinterpret_cast<UInt8 *>(g_cascade_name), g_cascade_name_len))
		abort();
	
	// create window, open camera stream, load cascade and create a buffer
	cvNamedWindow(g_window_name, CV_WINDOW_AUTOSIZE);
	CvCapture *camera = cvCreateCameraCapture (CV_CAP_ANY);
	if (!camera)
		abort();
	CvHaarClassifierCascade *cascade = (CvHaarClassifierCascade *)cvLoad(g_cascade_name, 0, 0, 0);
	if (!cascade)
		abort();
	CvMemStorage *storage = cvCreateMemStorage(0);
	assert(storage);
	
	// get an initial frame and duplicate it for later work
	IplImage *current_frame = cvQueryFrame(camera);
	IplImage *draw_image    = cvCreateImage(cvSize(current_frame->width, current_frame->height), IPL_DEPTH_8U, 3);
	IplImage *gray_image    = cvCreateImage(cvSize(current_frame->width, current_frame->height), IPL_DEPTH_8U, 1);
	IplImage *small_image   = cvCreateImage(cvSize(current_frame->width/scale, current_frame->height/scale), IPL_DEPTH_8U, 1);
	assert (current_frame && gray_image && draw_image);
	
	// as long as there are images ...
	while (current_frame = cvQueryFrame(camera)) {
		// convert to gray and downsize
		cvCvtColor(current_frame, gray_image, CV_BGR2GRAY);
		cvResize(gray_image, small_image, CV_INTER_LINEAR);
		
		// detect
		CvSeq* eyes = cvHaarDetectObjects(small_image, cascade, storage, 1.1, 2,
		                                  CV_HAAR_DO_CANNY_PRUNING, cvSize(30, 30));
		
		// draw areas
		cvFlip(current_frame, draw_image, 1);
		
		for (int i = 0; i < (eyes ? eyes->total : 0); i++) {
			CvRect *r = (CvRect *)cvGetSeqElem(eyes, i);
			CvPoint center;
			center.x = cvRound((small_image->width - r->width*0.5 - r->x) *scale);
			center.y = cvRound((r->y + r->height*0.5)*scale);
			int radius = cvRound((r->width + r->height)*0.25*scale);
			cvCircle(draw_image, center, radius, CV_RGB(0,255,0), 3, 8, 0);
		}
		
		// just show the image
		cvShowImage(g_window_name, draw_image);
		
		// wait a tenth of a second for keypress and window drawing
		int key = cvWaitKey(refreshDelay);
		if (key == 'q' || key == 'Q')
			break;
	}
	
	return 0;
}
