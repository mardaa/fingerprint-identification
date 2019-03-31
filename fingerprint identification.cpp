#include <stdio.h>
#include <math.h>
//#include <Python.h>
#include <dirent.h> 
#include <string.h>
//#define MAX_IMG_INDEX 97
#define TARGET_INDEX 2
#define IMG_DIR "/home/marda/Desktop/PNG/"
#define NUMBER_OF_MINUTIAE 25
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"


typedef struct 
{
int xp[NUMBER_OF_MINUTIAE];
int yp[NUMBER_OF_MINUTIAE];
int anglep[NUMBER_OF_MINUTIAE];
char typee[NUMBER_OF_MINUTIAE];
}FEATURE_T;

// global declarations
int MAX_IMG_INDEX = 97;
double DISTANCE_DB[97];
FEATURE_T* FEATURE_DB = NULL;
FEATURE_T* TARGET = NULL;
FEATURE_T* feature=NULL;
//PyObject *pName, *pFunc, *pArgs, *pModule,*pmModule, *pimModule, *plModule, *pValue, *pgValue, *psValue, *item;
int validity_checker;

// function declaration
double get_dis(FEATURE_T fy, FEATURE_T fz);
unsigned int get_min_dist_index(double v[MAX_IMG_INDEX]);
FEATURE_T Get_feature(char *n);
void load_FEATURE_DB();
int index_match(FEATURE_T TARGET);
//int viewImage( PyObject *pnValue);
//int pixels(int c, int d);
float Get_localOrnt(Mat block , int blockcenteri , int blockcenterj);
void thinning(cv::Mat& im);
void thinningIteration(cv::Mat& im, int iter);

//main function
int main(int argc, char *argv[])
{
	//Py_SetProgramName(argv[0]); 
	//Py_Initialize();
	int a;	
	FEATURE_DB = malloc(42949*sizeof(FEATURE_T));
  	TARGET = malloc(10*sizeof(FEATURE_T));
 	feature = (FEATURE_T *)malloc(429*sizeof(FEATURE_T));
	if (FEATURE_DB == 0)
		{
			printf("ERROR: Out of memory\n");
			return 1;
		}
	load_FEATURE_DB();
	//assigment of the TARGET image index
	*TARGET = FEATURE_DB[TARGET_INDEX];
	a = index_match(*TARGET);
	if (a == TARGET_INDEX)
		printf("FINGERPRINT INDEX MATCHING SUCCESSFUL!!!!!!!!!!!!!! \n");
	else
		printf("No match found in the database \n");
	free(feature);
	free(TARGET);
	free(FEATURE_DB);
	//Py_Finalize();
return 0;
}

// function definitions: function#1
void load_FEATURE_DB()
{
	char *path=IMG_DIR;
	DIR *d;
	char fn[40];
	int i=0;
	int oo;
	struct dirent *dir;
	struct stat stbuf;
	d = opendir(IMG_DIR);
	if (d) {
		for(i=0;i<MAX_IMG_INDEX;i++) {
			if((dir = readdir(d))!=NULL) {
				validity_checker=0;
				oo=i+1;
				strcpy(fn,path);
				strcat(fn,dir->d_name);
				lstat(fn,&stbuf);
				if(S_ISREG(stbuf.st_mode)){
					printf("                    LOADING IMAGE NUMBER: %d \n",oo);
					//printf("file_name == %s\n",fn);
					FEATURE_DB[i] = Get_feature(fn);
					if (validity_checker == 1){
						--i; // discards invalid fingerprint
						MAX_IMG_INDEX = MAX_IMG_INDEX - 1;
					}
				}
				else {
					--i;
				}
			}		
		}
		closedir(d);
	}
}

// function#2
FEATURE_T Get_feature(char *n)
{
//******************************************************************image read from a folder
	Mat gray,blur,num,den,bw,norm;
	Mat img=imread("/home/marda/Desktop/PNG/1_1.png");
//*************************************************************************Gray
	cvtColor(img, gray, CV_RGB2GRAY); 
//***************************************************************************normalization
	gray.convertTo(gray, CV_32F, 1.0/255.0);
	cv::GaussianBlur(gray, blur, Size(0,0), 2, 2);
	num = gray - blur;
	cv::GaussianBlur(num.mul(num), blur, Size(0,0), 20, 20);
	cv::pow(blur, 0.5, den);
	gray = num / den;
	cv::normalize(gray, norm, 0.0, 1.0, NORM_MINMAX, -1); 
//***************************************************************************orientation
	int blockSize=norm.cols/15-1; // defining block size; img.cols is 388 so blocksize is 17, tried by 10 output not good
	//if (!blockSize%2) 
		//blockSize+=1;
	printf("blocksize %d\n",blockSize);
	Mat ornt = Mat::zeros((norm.rows),(norm.cols),CV_32FC1);
	orntdraw=img.clone();
	printf("this is z orientation number of rows and cols %d and %d \n",ornt.rows,ornt.cols);

	float r=blockSize;
	int ii=0, jj=0; 
	int blockcenter_i, blockcenter_j;
	for (int i=0;i< norm.rows-blockSize;i+= blockSize) {
		for (int j=0;j< norm.cols-blockSize;j+= blockSize) { // dividing image into blocks and accessing them
			blockcenter_i = i + (blockSize/2);
			blockcenter_j = j + (blockSize/2);
			// printf("block center x,y= %d,%d   ",blockcenter_i, blockcenter_j);
			float a=Get_localOrnt(norm(Rect(j,i,blockSize,blockSize)) , blockcenter_i , blockcenter_j);
			float dx, px, py;
			int x=blockcenter_i;
			int y=blockcenter_j;            
			dx=(1/r)*x*cos(a) + (1/r)*y*sin(a);
			float dy=(1/r)*y*cos(a) - (1/r)*x*sin(a);
			//dx=cos(2*a);
			//float dy=sin(2*a);
			ornt.at<float>(blockcenter_i,blockcenter_j) = a;
			ii+=ii;
			jj+=jj;
			px = (1/1)*(blockcenter_i - ((blockSize/2)/tan(a)));
			py = (1/1)*(blockcenter_j - sqrt(pow(((blockSize/2)/sin(a)),2) - pow((blockcenter_i - px),2)));
			//cv::line(img,cv::Point(x,y),cv::Point(px,py),Scalar(0,0,255),1,CV_AA);
			cv::line(orntdraw,cv::Point(x,y),cv::Point(x+dx,y+dy),Scalar(0,0,255),1,CV_AA);
		}
	}
	imshow("oriented image",orntdraw);

//***************************************************************************ridge frequency
	int d,k,pcount=0,ppos[32],val=i+j*img.rows;//int val;
	float u,v,xsig[32],pmax,pmin,pfreq, Ofreq[val];
	double xsig[32]; //cv::Mat gx = cv::Mat::zeros(block.rows, block.cols, CV_16S); 
	for (int i=0;i< img.rows-blockSize;i+= blockSize) {
		for (int j=0;j< img.cols-blockSize;j+= blockSize) { // dividing image into blocks and accessing them
			blockcenter_i = i + (blockSize/2);
			blockcenter_j = j + (blockSize/2);
			for(k=0;k<32;k++) {
				xsig[k]=0.0;
				for(d=0;d<16;d++) {
					u = blockcenter_i + (d-blockSize/2)*cos(ornt.at<float>(blockcenter_i,blockcenter_j))+(k-16)*sin(ornt.at<float>(blockcenter_i,blockcenter_j));
					v = blockcenter_j + (d-blockSize/2)*sin(ornt.at<float>(blockcenter_i,blockcenter_j))+(16-k)*cos(ornt.at<float>(blockcenter_i,blockcenter_j));  
					xsig[k]+=(norm.at<float>(u,v));
				}
				xsig[k]/=16; 
				//printf("xsig[%d] ==%f      ",k,xsig[k]); 
			}
			//printf("val = %d    ",val);
			pmax = xsig[0];
			pmin = xsig[0];
			for(k=0;k<32;k++){
				if(xsig[k]<=pmin)
					pmin=xsig[k];
				if(xsig[k]>=pmax)
				pmax=xsig[k];
			}
			if((pmax-pmin)>64) {
				for(k=0;k<32;k++){
					if((xsig[k-1]<xsig[k])&&(xsig[k]>=xsig[k+1]))
					ppos[pcount++]=k;
				}
			}
			//pfreq=0.0;
			if(pcount>=2){
				for(k=0;k<pcount-1;k++){
					pfreq+=ppos[k+1]-ppos[k];
					pfreq/=pcount-1;
				}
			}
			//printf("pfreq %f   ",pfreq);
			if(pfreq >30)
				Ofreq[val]=0.0;
			else if(pfreq<2)
				Ofreq[val]=0.0;
			else
				Ofreq[val] =1/pfreq;
			printf("Ofreq %d == %d    ",val,Ofreq);			
	   
		}
	}
//***************************************************************************gaborfilter
	Mat img1, dest;
	img1=img.clone();
	int ks=ornt.rows - 1, hks = (ks-1)/2, phi= 0.55*CV_PI; // ks=21, hks=10 //printf("orntrows=%d \n",ks);
	cv::Mat kernel(ks,ks, CV_32F);
	double sigma = 5.0, x_theta, y_theta, del = 2.0/(ks-1), lmbd = 50;
	for (int y=-hks; y<=hks; y++) {
		for (int x=-hks; x<=hks; x++) {
			x_theta = x*del*cos(ornt.at<float>(hks+y,hks+x))+y*del*sin(ornt.at<float>(hks+y,hks+x));
			y_theta = -x*del*sin(ornt.at<float>(hks+y,hks+x))+y*del*cos(ornt.at<float>(hks+y,hks+x));
			kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + phi);
		}
	}
	cv::namedWindow("Process window", 1);
	img1.convertTo(img1, CV_32F, 0.7/255, 0);
	cv::filter2D(img1, dest, CV_32F, kernel);
	cv::imshow("Process window", dest);
//***************************************************************************binarization
	cv::threshold(img, bw, 180, 255, CV_THRESH_BINARY);
	imshow("binarized",bw);
//***************************************************************************thinning
	thinning(bw);
	cv::imshow("src", src);
	cv::imshow("dst", bw);
//******************************************************************PixelAccessing & MinuteaExtraction
	int b,x,y,a,c,d,e,f,gg,h,CN,w,q,angle;
	int count=0;
	w=0;
	q=0;
	for (y=1; y<100; y++){
		for (x=1; x<90; x++){
			if (pixels(x,y)==255){
				a=abs(pixels(x,y+1)-pixels(x+1,y+1));
				b=abs(pixels(x+1,y+1)-pixels(x+1,y));
				c=abs(pixels(x+1,y)-pixels(x+1,y-1));
				d=abs(pixels(x+1,y-1)-pixels(x,y-1));
				e=abs(pixels(x,y-1)-pixels(x-1,y-1));
				f=abs(pixels(x-1,y-1)-pixels(x-1,y));
				gg=abs(pixels(x-1,y)-pixels(x-1,y+1));
				h=abs(pixels(x-1,y+1)-pixels(x,y+1));
	  			CN=0.5 * (a+b+c+d+e+f+gg+h)/255;
				if (CN==1) {  // if termination is found....
					if (pixels(x,y+1) == 255)
						angle=270;
					else if (pixels(x+1,y+1) == 255)
						angle=225;
					else if (pixels(x+1,y) == 255)
						angle=180;
					else if (pixels(x+1,y-1) ==255)
						angle=135;
					else if (pixels(x,y-1)==255)
						angle=90;
					else if (pixels(x-1,y-1)==255)
						angle=45;
					else if (pixels(x-1,y)==255)
						angle=0;
					else if (pixels(x-1,y+1)==255)
						angle=315;
					if (count < NUMBER_OF_MINUTIAE) { //saving image minutiae to feature database
						feature->xp[count]=x;
						feature->yp[count]=y;
						feature->anglep[count]=angle;
						feature->typee[count]='T';
					}
                        		else   break;
			 		//printf("Termination found at %d, %d and angle=%d \n",x,y,angle);   
					count=count+1;
					w=w+1;
				}
		       		else if (CN==3) { //if bifurcation is found
			  		if ((pixels(x,y+1) == 255) && (pixels(x-1,y+1)!= 255)&&(pixels(x+1,y+1)!=255))
			    			angle=0;
		           		else if ((pixels(x-1,y+1)==255)&&(pixels(x,y+1)!=255) &&(pixels(x+1,y+1)!=255)&&(pixels(x-1,y)!=255)&&(pixels(x-1,y-1)!=255))
			     			angle=45;
			   		else if ((pixels(x-1,y+1)!=255)&&(pixels(x-1,y-1)!=255)&&(pixels(x-1,y)==255))
						angle=90;
			   		else if ((pixels(x-1,y+1)!=255)&&(pixels(x,y-1)!=255) &&(pixels(x+1,y-1)!=255)&&(pixels(x-1,y)!=255)&&(pixels(x-1,y-1)==255))
						angle=135;
			   		else if ((pixels(x+1,y+1)!=255)&&(pixels(x-1,y-1)!=255)&&(pixels(x,y-1)==255))
						angle=180;
			   		else if ((pixels(x+1,y+1)!=255)&&(pixels(x,y-1)!=255) &&(pixels(x+1,y)!=255)&&(pixels(x+1,y-1)!=255)&&(pixels(x-1,y-1)!=255))
						angle=225;
			   		else if ((pixels(x+1,y+1)!=255)&&(pixels(x+1,y-1)!=255)&&(pixels(x+1,y)==255))
						angle=270;
			 		else if ((pixels(x-1,y+1)!=255)&&(pixels(x,y+1)!=255) &&(pixels(x+1,y+1)==255)&&(pixels(x+1,y)!=255)&&(pixels(x+1,y-1)!=255))
						angle=315;
					if (count < NUMBER_OF_MINUTIAE)  { //saving image minutiaes to feature database
						feature->xp[count]=x;
						feature->yp[count]=y;
						feature->anglep[count]=angle;
						feature->typee[count]='B';
					}
					else   break;
			  		count=count+1;
					q=q+1;
					//printf("Bifurcation found at %d, %d and angle=%d \n",x,y,angle);   
				}
		 	}
		}
	}
	printf("Number of termination found: %d \n", w);
	printf("Number of Bifurcation found: %d \n", q);
	printf("Number of minutiaes found: %d \n", count);
	if (count == 0){
		printf("No minuteas found, INVALID FINGERPRINT !!!!");
		validity_checker=1;
	}
	
	
	return *feature;
}

//function#3
float Get_localOrnt(Mat block , int blockcenteri , int blockcenterj)
{
	float vx=0, vy=0, theta;
	Mat ang,mag;
	cv::Mat gx = cv::Mat::zeros(block.rows, block.cols, CV_16S); 
	cv::Mat gy = cv::Mat::zeros(block.rows, block.cols, CV_16S);
	cv::Sobel(block,gx,CV_32FC1,1,0,7);
	cv::Sobel(block,gy,CV_32FC1,0,1,7);

	for (int u=(blockcenteri-(block.rows/2)); u<(blockcenteri+(block.rows/2)); u++) {
		for (int v=(blockcenterj-(block.rows/2)); v<(blockcenterj+(block.rows/2)); v++) {
			vx+= 2*(gx.at<float>(u,v))*(gy.at<float>(u,v));
			vy+= pow(gx.at<float>(u,v),2)-pow(gy.at<float>(u,v),2);
		}
	}

	if (vx==0)
		theta=0.5*CV_PI;
	else
		theta = 0.5*atan(vy/vx);
	return theta;
}

//function#4
void thinningIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
//function#5
void thinning(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    } 
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}

//function#6
int index_match(FEATURE_T TARGET)
{
	int l;
	for (l=0; l<MAX_IMG_INDEX; l++) { // for all features of all the images in the database	
		DISTANCE_DB [l] = get_dis(FEATURE_DB[l] , TARGET);
	}
	return get_min_dist_index(DISTANCE_DB);
}

//function#7
double get_dis( FEATURE_T fy, FEATURE_T fz)
{
	int i,j,count;
	double distance_t;
	double dif_ang,dif_dis,xx;
	double sum_dis=0;
	for (i=0; i<NUMBER_OF_MINUTIAE; i++){
		count=0;
		xx=100000; //large number which is possibly less than the first dif_dis 
		for (j=0; j<NUMBER_OF_MINUTIAE; j++)
		{
			if (fz.typee[i] == fy.typee[j]) { // calculate if only it has same minutea type		
				count=count+1;
				dif_ang = abs (fz.anglep[i] - fy.anglep[j]);
				if (dif_ang > 180 && dif_ang<=360)
					dif_ang = 360 - (fz.anglep[i] - fy.anglep[j]);
				dif_dis = sqrt(pow((fz.xp[i] - fy.xp[j]), 2) + pow((fz.yp[i] - fy.yp[j]), 2));	
		// calculate minimum of all distance between 1 minutea in fz(input/TARGET) and all minutea in fy(template)
				if (dif_dis < xx)
					xx=dif_dis;
			}			
  		}
		// calculates sum of all minimum distances of each minutea of fy & fz
		if (count == 0)
			xx=dif_dis; //assigning xx 0 in case if there are only either bifurcatin or termination minuteas in a fingerprint
		sum_dis = sum_dis + xx;
	}
	distance_t = sum_dis;
	return distance_t;
}

//function#8
unsigned int get_min_dist_index(double v[MAX_IMG_INDEX])
{
	int k, match_index=0;
	double current_min_index = v[0];
	double r0=0; // since this is trial and we take sample images from same folder, error is 0 so the treshold has to be set to 0 or number close to 0
	for (k=0; k<MAX_IMG_INDEX; k++)
	{
		printf("the distance of image index %d to the taget index == %f \n", k, v[k]);
		if (v[k] <= r0 && v[k] < current_min_index)
		{
			current_min_index = v[k];
			match_index = k;
		}
	
	}
	printf("the found matching index is===%d \n", match_index);
	return match_index;
}


