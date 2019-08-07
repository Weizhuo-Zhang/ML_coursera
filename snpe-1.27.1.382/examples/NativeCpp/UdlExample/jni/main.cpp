//==============================================================================
//
//  Copyright (c) 2015-2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//
// This file contains an example application that loads and executes a neural
// network using the SNPE C++ API and saves the layer output to a file.
// Inputs to and outputs from the network are conveyed in binary form as single
// precision floating point values.
//
// Description of the functionality in this example:
//   1. Load the DL container file
//   2. Create a SNPE network instance from the loaded container.
//      2a. Print out the SNPE library version and version information
//          that is stored in the DNN model.
//   3. Create the input tensor that conveys inputs into the network.
//   4. For each input file in the input file listing:
//      4a. Load the file contents into the input tensor
//      4b. Execute the network with the input tensor
//      4c. Save each of the network output tensors to a file
//

#include <getopt.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <numeric>
#include <algorithm>
#include <cerrno>
#include <cstdlib>

#include "SNPE/SNPE.hpp"
#include <ctime>
#include <chrono>

#include "SNPE/SNPEFactory.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DiagLog/IDiagLog.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"

#include "MyUdlLayers.hpp"
#include "udlMyCustomScaleLayer.hpp"

static std::vector<float> LoadInputFile(const std::string& inputFile);
static std::vector<unsigned char> LoadNV21InputFile(const std::string& inputFile);
static void SaveITensor(const std::string& path, const zdl::DlSystem::ITensor* tensor);
static void ProcessCommandLine(int argc, char** argv);
static void ShowHelp(void);
static bool EnsureDirectory(const std::string& dir);

// Command line settings
static std::string ContainerPath;
static std::string InputFileListPath;
static std::string OutputDir = "./output/";
static bool DebugNetworkOutputs;
static zdl::DlSystem::Runtime_t Runtime = zdl::DlSystem::Runtime_t::CPU;
static bool isInputNv21EncodingType = false;


namespace {
// Helper for splitting tokenized strings
template <typename Container>
Container& split(
  Container&                                 result,
  const typename Container::value_type&      s,
  typename Container::value_type::value_type delimiter
)
{
  result.clear();
  std::istringstream ss( s );
  while (!ss.eof())
  {
    typename Container::value_type field;
    getline( ss, field, delimiter );
    if (field.empty()) continue;
    result.push_back( field );
  }
  return result;
}

void PrintErrorStringAndExit(const std::string& err) {
   std::cerr << err;
   std::exit(EXIT_FAILURE);
}

void PrintErrorStringAndExit() {
   const char* const errStr = zdl::DlSystem::getLastErrorString();
   std::cerr << errStr << std::endl;
   std::exit(EXIT_FAILURE);
}

} // ns annonymous

namespace udlexample {
zdl::DlSystem::IUDL* MyUDLFactory(void* cookie, const zdl::DlSystem::UDLContext* c) {
   if (!c) return nullptr;
   if (cookie != (void*)0xdeadbeaf) {
       PrintErrorStringAndExit("MyUDLFactory cookie should be 0xdeadbeaf");
   }

   // Check the layer type
   const void* blob = c->getBlob();
   size_t size = c->getSize();
   myudl::CommonLayerParams params;
   if (!blob) {
      PrintErrorStringAndExit("Received null blob");
   }
   if(!myudl::ParseCommonLayerParams(blob, size, params)) {
       PrintErrorStringAndExit("Failed to parse common layer params");
   }
   switch(params.type) {
       case myudl::MY_CUSTOM_SCALE_LAYER:
           return new myudl::UdlMyCustomScale(*c);
       default:
           PrintErrorStringAndExit("Unknown layer type");
   }
   return nullptr;
}
} // ns udlexample

int main(int argc, char** argv)
{
   ProcessCommandLine(argc, argv);
   

   // Open the DL container that contains the network to execute.
   std::unique_ptr<zdl::DlContainer::IDlContainer> container;
   container = zdl::DlContainer::IDlContainer::open(ContainerPath);
   if (!container) {
      PrintErrorStringAndExit();
   }

   // Create an instance of the SNPE network from the now opened container.
   // The factory functions provided by SNPE allow for the specification
   // of which layers of the network should be returned as output and also
   // if the network should be run on the CPU or GPU.
   std::unique_ptr<zdl::SNPE::SNPE> snpe;
   // The runtime availability API allows for runtime support to be queried.
   // If a selected runtime is not available, we will issue a warning and continue,
   // expecting the invalid configuration to be caught at SNPE network creation.
   if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(Runtime)) {
         std::cout << "The selected runtime is not available on this platform. Continue anyway ";
         std::cout << "to observe the failure at network creation time." << std::endl;
   }

   // 0xdeadbeaf to test cookie
   zdl::DlSystem::UDLBundle udlBundle;
   udlBundle.cookie = (void*)0xdeadbeaf;
   udlBundle.func = udlexample::MyUDLFactory;
   zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
   snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessor(Runtime)
       .setUdlBundle(udlBundle)
       .setDebugMode(DebugNetworkOutputs)
       .build();
   if (!snpe) {
      PrintErrorStringAndExit();
   }
   std::cout << std::string(79, '-') << "\n";

   // Configure logging output and start logging
   auto logger_opt = snpe->getDiagLogInterface();
   if (!logger_opt) throw std::runtime_error("SNPE failed to obtain logging interface");
   auto logger = *logger_opt;
   auto opts = logger->getOptions();
   opts.LogFileDirectory = OutputDir;
   if(!logger->setOptions(opts)) {
      std::cerr << "Failed to set options" << std::endl;
      return EXIT_FAILURE;
   }
   if (!logger->start()) {
      std::cerr << "Failed to start logger" << std::endl;
      return EXIT_FAILURE;
   }

   // Print out version information about the model and SNPE library
   std::cout << "SNPE v" << zdl::SNPE::SNPEFactory::getLibraryVersion().toString() << "\n";

   std::cout << std::string(79, '-') << "\n";

   // Open the input file listing and for each input file load its contents
   // into a SNPE tensor, execute the network with the input and save
   // each of the returned output tensors to a file.
   std::ifstream fileListStream(InputFileListPath);
   if (!fileListStream)
   {
      std::cerr << "Failed to open input file: " << InputFileListPath << "\n";
      std::exit(EXIT_FAILURE);
   }

   size_t num = 0;
   std::string fileLine;

   while (std::getline(fileListStream, fileLine))
   {
      if (fileLine.empty()) continue;
      zdl::DlSystem::TensorMap outputTensorMap;
      const auto &strList_opt = snpe->getInputTensorNames();
      if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
      const auto &strList = *strList_opt;

      if (strList.size() == 1)
      {
         // If the network has a single input, each line represents the input
         // file to be loaded for that input
         std::unique_ptr<zdl::DlSystem::ITensor> input;
         std::string filePath(fileLine);
         std::cout << "Processing DNN Input: " << filePath << "\n";

         if(!isInputNv21EncodingType)
         {
            std::vector<float> inputVec = LoadInputFile(filePath);

            // Create an input tensor that is correctly sized to hold the input
            // of the network. Dimensions that have no fixed size will be represented
            // with a value of 0.
            const auto &inputDims_opt = snpe->getInputDimensions(strList.at(0));
            if (!inputDims_opt) throw std::runtime_error("Error obtaining input dimensions");
            const auto &inputShape = *inputDims_opt;

            // Calculate the total number of elements that can be stored in the
            // tensor so that we can check that the input contains the expected
            // number of elements.
            size_t inputSize = std::accumulate(
               inputShape.getDimensions(), inputShape.getDimensions() + inputShape.rank(),
               1, std::multiplies<size_t>());

            if (inputVec.size() != inputSize)
            {
               std::cerr << "Size of input does not match network.\n"
                         << "Expecting: " << inputSize << "\n"
                         << "Got: " << inputVec.size() << "\n";
               std::exit(EXIT_FAILURE);
            }

            // With the input dimensions computed create a tensor to convey the input
            // into the network.
            input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

            // Copy the loaded input file contents into the networks input tensor.
            // SNPE's ITensor supports C++ STL functions like std::copy()
            std::copy(inputVec.begin(), inputVec.end(), input->begin());
         }
         else
         {
            std::vector<unsigned char> inputVec = LoadNV21InputFile(filePath);
            const auto &inputShape_opt = snpe->getInputDimensions(strList.at(0));
            if (!inputShape_opt) throw std::runtime_error("Failed to obtain input dimensions");
            const auto &inputShape = *inputShape_opt;
            // NV21 file size = (width*height*3)/2
            size_t NV21InputSize = (inputShape[0]*inputShape[1]*3)/2;
            if (inputVec.size() != NV21InputSize)
            {
               std::cerr << "Size of nv21 input file does not match network.\n"
                         << "Expecting: " << NV21InputSize << "\n"
                         << "Got: " << inputVec.size() << "\n";
               std::exit(EXIT_FAILURE);
            }

            // Specialized tensor factory for nv21
            input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape,
                                                                            inputVec.data(),
                                                                            NV21InputSize);
         }
         // Execute the network with the given single input.
         if(!snpe->execute(input.get(), outputTensorMap)) {
            PrintErrorStringAndExit();
         }
      }
      else
      {
         // The network has multiple inputs. Treat each line as a
         // space-separated list of input files.
         std::vector<std::string> filePaths;
         split(filePaths, fileLine, ' ');
         zdl::DlSystem::StringList inputTensorNames = snpe->getInputTensorNames();
         std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> inputs(inputTensorNames.size());
         zdl::DlSystem::TensorMap  inputTensorMap;
         if (filePaths.size() != inputTensorNames.size())
         {
            std::cerr << "Number of input files does not match the number of"
                      << " inputs to the network.\n"
                      << "Expecting: " << inputTensorNames.size() << "\n"
                      << "Got: " << filePaths.size() << "\n";
            std::exit(EXIT_FAILURE);
         }
         std::cout << "Processing DNN Input: " << filePaths[0] << "\n";
         for (size_t i = 0; i<inputTensorNames.size(); i++)
         {
            std::string filePath(filePaths[i]);
            std::string inputName(inputTensorNames.at(i));

            if(!isInputNv21EncodingType)
            {
               std::vector<float> inputVec = LoadInputFile(filePath);

               // Create an input tensor that is correctly sized to hold the input
               // of the network. Dimensions that have no fixed size will be represented
               // with a value of 0.
               const auto &inputShape_opt = snpe->getInputDimensions(inputName.c_str());
               if (!inputShape_opt) throw std::runtime_error("Failed to obtain input dimensions");
               const auto &inputShape = *inputShape_opt;
               // Calculate the total number of elements that can be stored in the
               // tensor so that we can check that the input contains the expected
               // number of elements.
               size_t inputSize = std::accumulate(
                  inputShape.getDimensions(), inputShape.getDimensions() + inputShape.rank(),
                  1, std::multiplies<size_t>());

               if (inputVec.size() != inputSize)
               {
                  std::cerr << "Size of input does not match network.\n"
                            << "Expecting: " << inputSize << "\n"
                            << "Got: " << inputVec.size() << "\n";
                  std::exit(EXIT_FAILURE);
               }

               // With the input dimensions computed create a tensor to convey the input
               // into the network.
               inputs[i] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);

               // Copy the loaded input file contents into the networks input tensor.
               // SNPE's ITensor supports C++ STL functions like std::copy()
               std::copy(inputVec.begin(), inputVec.end(), inputs[i]->begin());
               inputTensorMap.add(inputName.c_str(), inputs[i].get());
            }
            else
            {
               std::vector<unsigned char> inputVec = LoadNV21InputFile(filePath);
               const auto &inputShape_opt = snpe->getInputDimensions(inputName.c_str());
               if (!inputShape_opt) throw std::runtime_error("Failed to obtain input dimensions");
               const auto &inputShape = *inputShape_opt;
               // NV21 file size = (width*height*3)/2
               size_t NV21InputSize = (inputShape[0]*inputShape[1]*3)/2;
               if (inputVec.size() != NV21InputSize)
               {
                  std::cerr << "Size of nv21 input file does not match network.\n"
                            << "Expecting: " << NV21InputSize << "\n"
                            << "Got: " << inputVec.size() << "\n";
                  std::exit(EXIT_FAILURE);
               }

               // Specialized tensor factory for nv21
               inputs[i] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape,
                                                                                   inputVec.data(),
                                                                                   NV21InputSize);
               inputTensorMap.add(inputName.c_str(), inputs[i].get());
            }
         }
         // Execute the network using the multi-input API
         if(!snpe->execute(inputTensorMap, outputTensorMap)) {
            PrintErrorStringAndExit();
         }
      }

      // Save each output layer to the output directory with the path:
      // OutputDir/Result_N/LayerName.raw
      zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();
      std::for_each( tensorNames.begin(),
                     tensorNames.end(),
                     [&](const char* name)
                     {
                        std::ostringstream path;
                        path << OutputDir << "/"
                             << "Result_" << num << "/"
                             << name << ".raw";

                        auto tensorPtr = outputTensorMap.getTensor(name);
                        SaveITensor(path.str(), tensorPtr);
                     });
      num++;
   }

   return 0;
}

std::vector<float> LoadInputFile(const std::string& inputFile)
{
   std::ifstream in(inputFile, std::ifstream::binary);
   if (!in.is_open() || !in.good())
   {
      std::cerr << "Failed to open input file: " << inputFile << "\n";
      std::exit(EXIT_FAILURE);
   }

   in.seekg(0, in.end);
   int length = in.tellg();
   in.seekg(0, in.beg);

   std::vector<float> vec;
   vec.resize(length/sizeof(float));
   if (!in.read(reinterpret_cast<char*>(&vec[0]), length))
   {
      std::cerr << "Failed to read the contents of: " << inputFile << "\n";
      std::exit(EXIT_FAILURE);
   }

   return vec;
}

// NV21 file contains YUV pixels in byte format.
std::vector<unsigned char> LoadNV21InputFile(const std::string& inputFile)
{
   std::ifstream in(inputFile, std::ifstream::binary);
   if (!in.is_open() || !in.good())
   {
      std::cerr << "Failed to open input file: " << inputFile << "\n";
      std::exit(EXIT_FAILURE);
   }

   in.seekg(0, in.end);
   int length = in.tellg();
   in.seekg(0, in.beg);

   std::vector<unsigned char> vec;
   vec.resize(length);
   if (!in.read(reinterpret_cast<char*>(&vec[0]), length))
   {
      std::cerr << "Failed to read the contents of nv21 file: " << inputFile << "\n";
      std::exit(EXIT_FAILURE);
   }

   return vec;
}

static void
SaveITensor(const std::string& path, const zdl::DlSystem::ITensor* tensor)
{
   // Create the directory path if it does not exist
   auto idx = path.find_last_of('/');
   if (idx != std::string::npos)
   {
      std::string dir = path.substr(0, idx);
      if (!EnsureDirectory(dir))
      {
         std::cerr << "Failed to create output directory: " << dir << ": "
                   << std::strerror(errno) << "\n";
         std::exit(EXIT_FAILURE);
      }
   }

   std::ofstream os(path, std::ofstream::binary);
   if (!os)
   {
      std::cerr << "Failed to open output file for writing: " << path << "\n";
      std::exit(EXIT_FAILURE);
   }

   for ( auto it = tensor->cbegin(); it != tensor->cend(); ++it )
   {
      float f = *it;
      if (!os.write(reinterpret_cast<char*>(&f), sizeof(float)))
      {
         std::cerr << "Failed to write data to: " << path << "\n";
         std::exit(EXIT_FAILURE);
      }
   }
}

void ProcessCommandLine(int argc, char** argv)
{
   enum OPTIONS
   {
      OPT_HELP          = 0,
      OPT_CONTAINER     = 1,
      OPT_INPUT_LIST    = 2,
      OPT_OUTPUT_DIR    = 3,
      OPT_DEBUG_OUTPUTS = 4,
      OPT_USE_GPU       = 5,
      OPT_USE_AIP       = 101,
      OPT_USE_FXP_CPU       = 102,
      OPT_ENCODING_TYPE = 6,
      OPT_USE_DSP       = 7
   };

   // Create the command line options
   static struct option long_options[] = {
      {"help",                no_argument,          NULL,  OPT_HELP },
      {"container",           required_argument,    NULL,  OPT_CONTAINER },
      {"input_list",          required_argument,    NULL,  OPT_INPUT_LIST },
      {"output_dir",          required_argument,    NULL,  OPT_OUTPUT_DIR },
      {"debug",               no_argument,          NULL,  OPT_DEBUG_OUTPUTS },
      {"use_gpu",             no_argument,          NULL,  OPT_USE_GPU },
      {"encoding_type",       required_argument,    NULL,  OPT_ENCODING_TYPE },
      {"use_dsp",             no_argument,          NULL,  OPT_USE_DSP},
      {"use_aip",             no_argument,          NULL,  OPT_USE_AIP },
      {"use_fxp_cpu",      no_argument,          NULL,  OPT_USE_FXP_CPU },
      {NULL,                  0,                    NULL,  0 }
   };

   // Command line parsing loop
   int long_index =0;
   int opt= 0;
   while ((opt = getopt_long_only(argc, argv, "", long_options, &long_index )) != -1)
   {
      switch (opt)
      {
         case OPT_HELP:
            ShowHelp();
            std::exit(0);
            break;

         case OPT_CONTAINER:
            ContainerPath = optarg;
            break;

         case OPT_INPUT_LIST:
            InputFileListPath = optarg;
            break;

         case OPT_DEBUG_OUTPUTS:
            DebugNetworkOutputs = true;
            break;

         case OPT_OUTPUT_DIR:
            OutputDir = optarg;
            break;

         case OPT_USE_GPU:
            Runtime = zdl::DlSystem::Runtime_t::GPU;
            break;
         case OPT_USE_DSP:
            Runtime = zdl::DlSystem::Runtime_t::DSP;
            break;

         case OPT_USE_AIP:
            Runtime = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
            break;
  

         case OPT_USE_FXP_CPU:
            Runtime = zdl::DlSystem::Runtime_t::CPU_FIXED8_TF;
            break;
  
         case OPT_ENCODING_TYPE:
            if (std::strcmp("nv21", optarg) == 0)
            {
                isInputNv21EncodingType = true;
            }
            break;

         default:
            ShowHelp();
            std::exit(EXIT_FAILURE);
      }
   }

   if (ContainerPath.empty())
   {
      std::cerr << "Missing option: --container\n";
      ShowHelp();
      std::exit(EXIT_FAILURE);
   }

   if (InputFileListPath.empty())
   {
      std::cerr << "Missing option: --input_list\n";
      ShowHelp();
      std::exit(EXIT_FAILURE);
   }
}

void ShowHelp(void)
{
   std::cout
      << "\nDESCRIPTION:\n"
      << "------------\n"
      << "Example application demonstrating how to load and execute a neural network\n"
      << "using the SNPE C++ API.\n"
      << "\n\n"
      << "REQUIRED ARGUMENTS:\n"
      << "-------------------\n"
      << "  --container  <FILE>   Path to the DL container containing the network.\n"
      << "  --input_list <FILE>   Path to a file listing the inputs for the network.\n"
      << "\n\n"
      << "OPTIONAL ARGUMENTS:\n"
      << "-------------------\n"
      << "  --use_gpu             Use the GPU runtime for SNPE.\n"
      << "  --use_dsp             Use the DSP fixed point runtime for SNPE.\n"
      << "  --use_aip             Use the AIP fixed point runtime for SNPE.\n" 
  << "  --use_fxp_cpu         Use the CPU fixed point runtime for SNPE.\n" 
      << "  --debug               Specifies that output from all layers of the network\n"
      << "                        will be saved.\n"
      << "  --output_dir <DIR>    The directory to save output to. Defaults to ./output\n"
      << "  --encoding_type <VAL> Specifies the encoding type of input file. Valid settings are \"nv21\". \n"
      << "  --help                Show this help message.\n"
      << std::endl;
}

bool EnsureDirectory(const std::string& dir)
{
   auto i = dir.find_last_of('/');
   std::string prefix = dir.substr(0, i);

   if (dir.empty() || dir == "." || dir == "..")
   {
      return true;
   }

   if (i != std::string::npos && !EnsureDirectory(prefix))
   {
      return false;
   }

   int rc = mkdir(dir.c_str(),  S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
   if (rc == -1 && errno != EEXIST)
   {
      return false;
   }
   else
   {
      struct stat st;
      if (stat(dir.c_str(), &st) == -1)
      {
         return false;
      }

      return S_ISDIR(st.st_mode);
   }
}
