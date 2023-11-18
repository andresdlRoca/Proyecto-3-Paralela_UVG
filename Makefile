all: pgm.o	houghGlobal houghTranConst houghShared

houghGlobal:	houghGlobal.cu pgm.o
	nvcc houghGlobal.cu pgm.o -o houghGlobal -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

houghShared: houghShared.cu pgm.o
	nvcc houghShared.cu pgm.o -o houghShared -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

houghTranConst: houghTranConst.cu pgm.o
	nvcc houghTranConst.cu pgm.o -o houghTranConst -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o
all: pgm.o	houghGlobal houghTranConst houghShared

houghGlobal:	houghGlobal.cu pgm.o
	nvcc houghGlobal.cu pgm.o -o houghGlobal -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

houghShared: houghShared.cu pgm.o
	nvcc houghShared.cu pgm.o -o houghShared -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

houghTranConst: HoughTranConst.cu pgm.o
	nvcc HoughTranConst.cu pgm.o -o houghTranConst -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o
