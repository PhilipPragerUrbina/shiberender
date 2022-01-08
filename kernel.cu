
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <SDL.h>
#include <stdio.h>

#include <chrono>
#include <curand_kernel.h>
#include <fstream>
#include <string>
#include <sstream>

#include <algorithm>
#include <helper_functions.h> 
#include <filesystem>
#include <math.h>
#include"float3Helpers.cuh"




//obj loader

#define TINYOBJLOADER_IMPLEMENTATION

//traingulate
#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"



//##DOGERAY## by Phil














//settings:
//render dimensions
int SCREEN_WIDTH = 720;
int SCREEN_HEIGHT = 720;
//factor to scale window up for small resoultions
const int upscale = 1;










//create namespace for filesystem
namespace fs = std::filesystem;


//triangle struct
typedef struct
{

    float3 pos1;

    float3 pos3;


    float3 pos2;

    //color
    float3 col;

}tri;




//settings struct
typedef struct
{

    float3 lookat;
    float3 campos;

}config;










// global variables for keeping trak of objects
int objnum = 10000;
int texnum = 1;


//camera settings(class TODO)
float3 campos = { 0, 0, 2 };
float3 look = { 0, 0, 0 };


//Define cuda starter function 
cudaError_t StartVertexShader(int3* outputr, tri* allobjects, cudaTextureObject_t* texarray);




//2d to 1d conversion
__device__ int getw(int x, int y, int SCREEN_HEIGHT) {
    return x * SCREEN_HEIGHT + y;

}


//compute matrix
__device__ Matrix3x3_d lookAt(float3 from, float3 to)
{
    float3 tmp = make_float3(0, -1, 0);
    float3 forward = getNormalizedVec(from - to);
    float3 right = getCrossProduct(getNormalizedVec(tmp), forward);
    float3 up = getCrossProduct(forward, right);

    Matrix3x3_d camToWorld = getIdentityMatrix3x3();

    camToWorld.m_row[0].x = right.x;
    camToWorld.m_row[0].y = right.y;
    camToWorld.m_row[0].z = right.z;


    camToWorld.m_row[1].x = up.x;
    camToWorld.m_row[1].y = up.y;
    camToWorld.m_row[1].z = up.z;

    camToWorld.m_row[2].x = forward.x;
    camToWorld.m_row[2].y = forward.y;
    camToWorld.m_row[2].z = forward.z;


    camToWorld.m_row[3].x = from.x;
    camToWorld.m_row[3].y = from.y;
    camToWorld.m_row[3].z = from.z;


    return camToWorld;
}


//get pixel coordates from camera
__device__ float2 computePixelCoordinates(
    float3 worldPoint,
    Matrix3x3_d worldtocamera,
    float screenwidth,
    float screenheight,
    int imagewidth,
    int imageheight
)

{





    //multiply matrix
    float3 pcamera = multVecMatrix(worldtocamera, worldPoint);


    //get screeen coordinates
    float2 pscreen = { 0,0 };
    pscreen.x = pcamera.x / -pcamera.z;

    pscreen.y = pcamera.y / -pcamera.z;



    //check if visible

    if (abs(pscreen.x) > screenwidth || abs(pscreen.y) > screenheight) {
        return make_float2(0, 0);

    }

    // normalize
    float2 pnormalized = { 0,0 };
    pnormalized.x = (pscreen.x + screenwidth / 2) / screenwidth;

    pnormalized.y = (pscreen.y + screenheight / 2) / screenheight;


    float2 praster = { 0,0 };
    //convert to pixel coords
    praster.x = floor(pnormalized.x * imagewidth);
    praster.y = floor((1 - pnormalized.y) * imageheight);

    return praster;









    //from scratapixel. Does not seem to work well

    /*
    float nearClippingPlane =0.1 ;
    float inchToMm = 25.4;
    float focalLength = 200; // in mm
    float filmApertureWidth =100;
    float filmApertureHeight = 100;
    float t, r, l, b;
    float filmAspectRatio = filmApertureWidth / filmApertureHeight;
    float deviceAspectRatio = imageWidth / (float)imageHeight;

    t = ((filmApertureHeight * inchToMm / 2) / focalLength) * nearClippingPlane;
    r = ((filmApertureWidth * inchToMm / 2) / focalLength) * nearClippingPlane;

    // field of view (horizontal)
    float fov = 45;


    float xscale = 1;
    float yscale = 1;


        if (filmAspectRatio > deviceAspectRatio) {
            yscale = filmAspectRatio / deviceAspectRatio;
        }
        else {
            xscale = deviceAspectRatio / filmAspectRatio;
        }



    r *= xscale;
    t *= yscale;

    b = -t;
    l = -r;





    // point in camera space

   float3 pCamera = multVecMatrix(worldToCamera,pWorld);
    // convert to screen space
   float2 pScreen = { 0,0 };
    pScreen.x = nearClippingPlane * pCamera.x / -pCamera.z;
    pScreen.y = nearClippingPlane * pCamera.y / -pCamera.z;
    // now convert point from screen space to NDC space (in range [-1,1])
    float2 pNDC = { 0,0 };
    pNDC.x = 2 * pScreen.x / (r - l) - (r + l) / (r - l);
    pNDC.y = 2 * pScreen.y / (t - b) - (t + b) / (t - b);
    // convert to raster space and set point z-coordinate to -pCamera.z
    float2 pRaster = { 0,0 };
    pRaster.x = (pScreen.x + 1) / 2 * imageWidth;
    // in raster space y is down so invert direction
    pRaster.y = (1 - pScreen.y) / 2 * imageHeight;
    // store the point camera space z-coordinate (as a positive value)
    //pRaster.z = -pCamera.z;



    return pRaster;

    */

}


//draw point on screen from GPU
__device__ void point(int x, int y, float3 col, int SCREEN_HEIGHT, int SCREEN_WIDTH, int3* outputr) {


    //get array index
    int w = getw(x, y, SCREEN_HEIGHT);


    //check if on screen
    if (x < SCREEN_WIDTH - 1 && x > 0 && y < SCREEN_HEIGHT - 1 && y > 0) {

        //edit output
        outputr[w].x = col.x;
        outputr[w].y = col.y;
        outputr[w].z = col.z;

    }




}

//use brenshams line algorthm to draw a line
__device__ void drawline(int x0, int y0, int x1, int y1, int3* outputr, int SCREEN_HEIGHT, float3 col, int SCREEN_WIDTH)
{

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2; /* error value e_xy */

    for (;;) {  /* loop */
        point(x0, y0, col, SCREEN_HEIGHT, SCREEN_WIDTH, outputr);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }



}
//vertex shader
__global__ void VertexShader(int3* outputr, config settings, tri* b, cudaTextureObject_t* tex, int SCREEN_WIDTH, int SCREEN_HEIGHT, int number)
{



    //get object id
    int object = blockIdx.x * blockDim.x + threadIdx.x;


    if (object > number-1) { return; }

    //get object color
    float3 col = b[object].col * make3(100) + make3(20);



    

    //setup camera
    Matrix3x3_d cam = lookAt(settings.campos, settings.lookat);
    cam = inverse(cam);


    //first point
    float2 coor = computePixelCoordinates(b[object].pos1, cam, 2, 2, SCREEN_WIDTH, SCREEN_HEIGHT);
    if (coor.x == 0 || coor.y == 0) {
        return;

    }
    float2 v1c;


    float2 v2c;







    v1c = coor;



    //second poiint
    coor = computePixelCoordinates(b[object].pos2, cam, 2, 2, SCREEN_WIDTH, SCREEN_HEIGHT);
    if (coor.x == 0 || coor.y == 0) {
        return;

    }



    v2c = coor;


    //first line
    drawline(int(coor.x), int(coor.y), int(v1c.x), int(v1c.y), outputr, SCREEN_HEIGHT, col, SCREEN_WIDTH);






    coor = computePixelCoordinates(b[object].pos3, cam, 2, 2, SCREEN_WIDTH, SCREEN_HEIGHT);
    if (coor.x == 0 || coor.y == 0) {
        return;

    }




    //other lines
    drawline(coor.x, coor.y, v2c.x, v2c.y, outputr, SCREEN_HEIGHT, col, SCREEN_WIDTH);




    drawline(int(coor.x), int(coor.y), int(v1c.x), int(v1c.y), outputr, SCREEN_HEIGHT, col, SCREEN_WIDTH);







}



//get number of tris
int getnum(std::string File) {

    if (File.find("obj") != std::string::npos) {
        std::string inputfile = File;
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; // Path to material files

        tinyobj::ObjReader reader;

        if (!reader.ParseFromFile(inputfile, reader_config)) {
            if (!reader.Error().empty()) {
                std::cerr << "TinyObjReader: " << reader.Error();
            }
            exit(1);
        }

        if (!reader.Warning().empty()) {
            std::cout << "TinyObjReader: " << reader.Warning();
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        //  auto& materials = reader.GetMaterials();
        int num = 0;
        for (size_t s = 0; s < shapes.size(); s++) {
            // Loop over faces(polygon)
            std::cout << s;
            size_t index_offset = 0;
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {


                num++;

            }
        }


        std::cout << "Obj: ";

        return  num;
    }
    else {
        using namespace std;
        string myText;
        ifstream MyReadFile;
        // Read from the text file




        MyReadFile.open(File);








        int line = 0;
        // Use a while loop together with the getline() function to read the file line by line

        if (MyReadFile.is_open()) {

            while (getline(MyReadFile, myText)) {



                if (myText[0] == "/"[0])
                    continue;
                if (myText[0] == "*"[0]) {

                    continue;


                }
                //increment number
                line++;
            }


            // Close the file
            MyReadFile.close();
            return line + 1;
        }
    }

}

//get texture id from name
int gettexnum(std::string query, std::string* texpaths) {
    for (int i = 0; i < texnum; i++)
    {

        std::string tobeq = texpaths[i];
        std::transform(tobeq.begin(), tobeq.end(), tobeq.begin(), ::tolower);
        if (tobeq.find(query) != std::string::npos) {
            return i;
        }
    }
    return -1;
}

//read file
void read(std::string File, tri* b, std::string* texpaths) {


    if (File.find("obj") != std::string::npos) {

        std::string inputfile = File;
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = "./"; // Path to material files

        tinyobj::ObjReader reader;

        if (!reader.ParseFromFile(inputfile, reader_config)) {
            if (!reader.Error().empty()) {
                std::cerr << "TinyObjReader: " << reader.Error();
            }
            exit(1);
        }

        if (!reader.Warning().empty()) {
            std::cout << "TinyObjReader: " << reader.Warning();
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();


        // Loop over shapes
        for (size_t s = 0; s < shapes.size(); s++) {
            // Loop over faces(polygon)
            std::cout << s;
            size_t index_offset = 0;
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

                // Loop over vertices in the face.
                for (size_t v = 0; v < fv; v++) {
                    // access to vertex



                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                    if (v == 0) {
                        b[f].pos1.x = -attrib.vertices[3 * size_t(idx.vertex_index) + 0];

                        b[f].pos1.y = -attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                        b[f].pos1.z = -attrib.vertices[3 * size_t(idx.vertex_index) + 2];



                    }
                    else if (v == 1) {

                        b[f].pos2.x = -attrib.vertices[3 * size_t(idx.vertex_index) + 0];

                        b[f].pos2.y = -attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                        b[f].pos2.z = -attrib.vertices[3 * size_t(idx.vertex_index) + 2];


                    }
                    else if (v == 2) {


                        b[f].pos3.x = -attrib.vertices[3 * size_t(idx.vertex_index) + 0];

                        b[f].pos3.y = -attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                        b[f].pos3.z = -attrib.vertices[3 * size_t(idx.vertex_index) + 2];


                        b[f].col.x = attrib.colors[3 * size_t(idx.vertex_index) + 0];
                        b[f].col.y = attrib.colors[3 * size_t(idx.vertex_index) + 1];
                        b[f].col.z = attrib.colors[3 * size_t(idx.vertex_index) + 2];

                    }
                    else if (v == 3) {

                        printf("quad!!!!!!!!!!!!!!!!!!!");
                    }












                }
                index_offset += fv;


            }


        }

    }
    else {

        // Read from the text file
        //C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp


        using namespace std;
        string myText;
        ifstream MyReadFile;
        MyReadFile.open(File);








        int line = 0;
        // Use a while loop together with the getline() function to read the file line by line

        if (MyReadFile.is_open()) {

            while (getline(MyReadFile, myText)) {
                //what colum
                int colum = 0;
                stringstream s_stream(myText); //create string stream from the string

                //ignore comments
                if (myText[0] == "/"[0]) {

                    continue;
                }
                //read settings
                if (myText[0] == "*"[0]) {
                    continue;


                }
                while (s_stream.good()) {
                    string substr;
                    getline(s_stream, substr, ','); //get first string delimited by comma


                    //appl info
                    if (colum == 0) {
                        b[line].pos1.x = stof(substr);


                    }
                    else if (colum == 1) {

                        b[line].pos1.y = stof(substr);
                    }
                    else if (colum == 2) {

                        b[line].pos1.z = stof(substr);
                    }

                    else if (colum == 4) {

                        b[line].col.x = stof(substr);
                    }
                    else if (colum == 5) {

                        b[line].col.y = stof(substr);
                    }
                    else if (colum == 6) {

                        b[line].col.z = stof(substr);
                    }

                    else if (colum == 9) {

                        b[line].pos2.x = stof(substr);
                    }
                    else if (colum == 10) {

                        b[line].pos2.y = stof(substr);
                    }
                    else if (colum == 11) {

                        b[line].pos2.z = stof(substr);
                    }

                    else if (colum == 13) {
                        b[line].pos3.x = stof(substr);


                    }
                    else if (colum == 14) {

                        b[line].pos3.y = stof(substr);
                    }
                    else if (colum == 15) {

                        b[line].pos3.z = stof(substr);
                    }








                    colum++;
                }
                line++;
            }


            // Close the file
            MyReadFile.close();
        }


    }
}



//read and allocate textures
void readtextures(cudaTextureObject_t* texarray, std::string* texpaths) {


    //load textures and alloacate them
    for (int i = 0; i < texnum; i++)
    {
        unsigned char* hData = NULL;
        unsigned int width, height;
        char* imagePath = strcpy(new char[texpaths[i].length() + 1], texpaths[i].c_str());


        sdkLoadPPM4(imagePath, &hData, &width, &height);





        unsigned int sizee = width * height * sizeof(uchar4);


        // Allocate array and copy image data
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray* cuArray;
        cudaMallocArray(&cuArray,
            &channelDesc,
            width,
            height);
        cudaMemcpyToArray(cuArray,
            0,
            0,
            hData,
            sizee,
            cudaMemcpyHostToDevice);

        cudaTextureObject_t         tex;
        cudaResourceDesc            texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = cuArray;

        cudaTextureDesc             texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = true;
        texDescr.filterMode = cudaFilterModePoint;
        texDescr.addressMode[0] = cudaAddressModeWrap;
        texDescr.addressMode[1] = cudaAddressModeWrap;
        texDescr.addressMode[2] = cudaAddressModeWrap;
        texDescr.addressMode[3] = cudaAddressModeWrap;
        //  texDescr.readMode = cudaReadModeElementType;

        cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL);



        //add to array
        texarray[i] = tex;
        delete[] imagePath;
    }

}

//get number of textures
int getppmnum() {
    //get number of ppm textures
    std::string path = fs::current_path().string();
    int i = 0;
    for (const auto& entry : fs::directory_iterator(path)) {

        std::string newpath{ entry.path().u8string() };
        if (newpath.find("ppm") != std::string::npos || newpath.find("PPM") != std::string::npos) {

            std::cout << "Found: " << entry.path() << std::endl;

            i++;
        }


    }
    std::cout << i << " textures total" << std::endl;
    return i;

}

//get txture paths
void getppmpaths(std::string* things) {
    //add texture paths to array
    std::string path = fs::current_path().string();
    int i = 0;
    for (const auto& entry : fs::directory_iterator(path)) {

        std::string newpath{ entry.path().u8string() };
        if (newpath.find("ppm") != std::string::npos || newpath.find("PPM") != std::string::npos) {

            things[i] = newpath;
            i++;
        }


    }



}



float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}


int main(int argc, char* args[])
{
    //check args so that files can be opened directly



    std::cout << R"(

   _____ _    _ _____ ____  ______ _____  ______ _   _ _____  ______ _____  
  / ____| |  | |_   _|  _ \|  ____|  __ \|  ____| \ | |  __ \|  ____|  __ \  (c)
 | (___ | |__| | | | | |_) | |__  | |__) | |__  |  \| | |  | | |__  | |__) |
  \___ \|  __  | | | |  _ <|  __| |  _  /|  __| | . ` | |  | |  __| |  _  / 
  ____) | |  | |_| |_| |_) | |____| | \ \| |____| |\  | |__| | |____| | \ \ 
 |_____/|_|  |_|_____|____/|______|_|  \_\______|_| \_|_____/|______|_|  \_\   GPU rasterizer
                                                                            
                                                                            

                                      
)" << "\n";


    std::cout << "      V.0   by Philip Prager Urbina   2021" << std::endl;
    std::cout << "      Find on github for documentation: https://github.com/PhilipPragerUrbina/SHIBERENDER" << std::endl << std::endl << std::endl << std::endl;


    std::string filename;
    if (argc < 2)
    {
        return 0;
    }
    else
    {
        filename = args[1];
        std::cout << "Opening:" << filename << std::endl;
    }


    //get number of objects
    objnum = getnum(filename);
    std::cout << objnum << " tris" << std::endl;



    //create object array  TODO: vector
    tri* allobjects = new tri[objnum];



    //get number of textures
    texnum = getppmnum();

    //create texure paths array
    std::string* texpaths = new std::string[texnum];
    //get texture paths
    getppmpaths(texpaths);




    //read rts or obj file
    std::cout << "Parsing file:" << std::endl;
    read(filename, allobjects, texpaths);

    std::cout << "read" << std::endl;



    //create texure  array
    cudaTextureObject_t* textures = new cudaTextureObject_t[texnum];

    //read textures
    readtextures(textures, texpaths);
    std::cout << "textures read" << std::endl;





    SDL_Event event;


    bool quit = false;
    SDL_Window* window = NULL;
    SDL_Renderer* renderer;


    std::cout << "Opening Window:" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;




    int s = SCREEN_WIDTH * SCREEN_HEIGHT;
    int3* outr = new int3[s];


    int3* noutr = new int3[s];


    //Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
    }
    else
    {

        //Create window
        SDL_CreateWindowAndRenderer(SCREEN_WIDTH * upscale, SCREEN_HEIGHT * upscale, 0, &window, &renderer);
        SDL_SetWindowTitle(window,
            "SHIBERENDER");
        if (window == NULL)
        {
            printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        }
        else
        {






            while (!quit)
            {





                //start timer
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();




                cudaError_t cudaStatus = StartVertexShader(outr, allobjects, textures);
























                SDL_Texture* texture = NULL;
                void* pixels;
                Uint8* base;
                int pitch;


            
                unsigned int x;
           
                unsigned int y;
     
                texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);
                SDL_LockTexture(texture, NULL, &pixels, &pitch);
                for (x = 0; x < SCREEN_WIDTH; x++) {
                    for (y = 0; y < SCREEN_HEIGHT; y++) {

                        int w = x * SCREEN_HEIGHT + y;
                        base = ((Uint8*)pixels) + (4 * (y * SCREEN_WIDTH +x));
                        base[0] = outr[w].x;
                        base[1] = outr[w].y;
                        base[2] = outr[w].z;
                        base[3] = 255;
                    }


                   
                }
                SDL_UnlockTexture(texture);
                SDL_RenderCopy(renderer, texture, NULL, NULL);






                /*
                //display pixels from output

                for (int x = 0; x < SCREEN_WIDTH; x++) {
                    for (int y = 0; y < SCREEN_HEIGHT; y++) {

                        //calulate w from xa and y
                        int w = x * SCREEN_HEIGHT + y;
                        //set pixel color, clamp, and proccess samples
                        SDL_SetRenderDrawColor(renderer, clamp(outr[w].x, 0, 255), clamp(outr[w].y, 0, 255), clamp(outr[w].z, 0, 255), 255);



                        //here things are upscaled to bigger windows for smaller resolutions
                        SDL_RenderDrawPoint(renderer, x * upscale, y * upscale);
                        if (upscale > 1) {


                            for (int u = 0; u < (upscale); u++) {
                                SDL_RenderDrawPoint(renderer, x * (upscale)+u, y * (upscale));
                                for (int b = 0; b < (upscale); b++) {
                                    SDL_RenderDrawPoint(renderer, x * (upscale)+u, y * (upscale)+b);

                                }
                            }

                        }
















                    }
                }
                */
                //stop timer
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                //display info
                std::cout << '\r' << "d: " << 0 << " " << "Time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]  " << 1e+6 / std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " FPS       ";



                bool toexport = false;
                //get input

                //update window
                SDL_RenderPresent(renderer);




                //proccess input
                while (SDL_PollEvent(&event)) {
                    switch (event.type)
                    {
                    case SDL_QUIT:
                        //handle close
                        quit = true;
                        SDL_DestroyRenderer(renderer);
                        break;


                    case SDL_KEYDOWN:

                        switch (event.key.keysym.sym) {
                        case SDLK_RIGHT:

                            //move camera right
                            campos.x += 1;

                            //sameple iteration is reset with movement
                           //for a motion blur effect just dont reset iter with motion 

                            break;
                        case SDLK_LEFT:
                            campos.x -= 1;
                            //move camera left



                            break;
                        case SDLK_UP:
                            //move camera forward
                            campos.z -= 1;

                            break;
                        case SDLK_DOWN:
                            //back
                            campos.z += 1;

                            break;
                        case SDLK_w:
                            //up
                            campos.y -= 0.5;

                            break;
                        case SDLK_s:
                            //down
                            campos.y += 0.5;

                            break;




                        case SDLK_SPACE:

                            toexport = true;
                            break;
                        case SDLK_ESCAPE:
                            //handle exit throug escape
                            quit = true;
                            SDL_DestroyRenderer(renderer);
                            break;

                        default:
                            break;
                        }
                    }
                }

                //export screenshot
                if (toexport == true) {

                    toexport = false;
                    //export image
                    SDL_Surface* sshot = SDL_CreateRGBSurface(0, SCREEN_WIDTH, SCREEN_HEIGHT, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000);
                    SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_ARGB8888, sshot->pixels, sshot->pitch);
                    char* patherchar;
                    std::string pather = filename + ".bmp";


                    patherchar = &pather[0];

                    SDL_SaveBMP(sshot, patherchar);
                    SDL_FreeSurface(sshot);
                    std::cout << "\n" << "exported image:" << filename << ".bmp" << "\n";
                }















            }



        }
    }




    //delete dynamic arrays

    delete[] allobjects;
    delete[] textures;

    //reset gpu
    cudaDeviceReset();
    //close window
    SDL_DestroyWindow(window);

    //Quit SDL subsystems
    SDL_Quit();
    //exit program
    return 0;

}



//this function starts the render kernel
cudaError_t StartVertexShader(int3* outputr, tri* allobjects, cudaTextureObject_t* texarray)
{
















    //set up settings values
    config settings;


    settings.campos = campos;
    settings.lookat = look;



    //calculate output size
    int size = SCREEN_WIDTH * SCREEN_HEIGHT;




    //placeholder pointers

    int3* dev_outputr = 0;
    tri* dev_allobjects = 0;
    cudaTextureObject_t* dev_texarray = 0;


    //set up error status
    cudaError_t cudaStatus;


    //get device
    int device = -1;
    cudaGetDevice(&device);


    // Allocate GPU buffers.
    cudaStatus = cudaMalloc((void**)&dev_outputr, size * sizeof(int3));
    cudaStatus = cudaMalloc((void**)&dev_allobjects, objnum * sizeof(tri));
    cudaStatus = cudaMalloc((void**)&dev_texarray, texnum * sizeof(cudaTextureObject_t));







    cudaStatus = cudaMemcpy(dev_allobjects, allobjects, objnum * sizeof(tri), cudaMemcpyHostToDevice);
    cudaMemPrefetchAsync(dev_allobjects, objnum * sizeof(tri), device, NULL);


    cudaStatus = cudaMemcpy(dev_texarray, texarray, texnum * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    cudaMemPrefetchAsync(dev_texarray, texnum * sizeof(cudaTextureObject_t), device, NULL);


    //calulate blocks and threads




    dim3 threadsPerBlock(16, 1);

    dim3 numBlocks(objnum - 1 / threadsPerBlock.x, 1);



    VertexShader << <numBlocks, threadsPerBlock >> > (dev_outputr, settings, dev_allobjects, dev_texarray, SCREEN_WIDTH, SCREEN_HEIGHT, objnum);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    cudaDeviceSynchronize();

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.


    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(outputr, dev_outputr, size * sizeof(int3), cudaMemcpyDeviceToHost);



    //free memory to avoid filling up vram

    cudaFree(dev_allobjects);

    cudaFree(dev_texarray);

Error:
    //just in case
    cudaFree(dev_outputr);




    return cudaStatus;
}

