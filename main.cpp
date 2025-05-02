#include "llama_executor.h"
#include <iostream>

void callBack(const std::string &responsePiece){
    std::cout << responsePiece << std::endl;
}

int main(int argc, char** argv) {

    std::string model_path = "H:\\workspace\\llama.cpp\\build\\bin\\Debug\\dolphin3b.gguf";
    int ngl = 99, n_ctx = 2048;

    LlamaExecutor executor;
    executor.setOptions(model_path, ngl, n_ctx);
    executor.setCallBackFunction(callBack);
    executor.chat("hello how are you?");

    
    // std::string text1 = "Today is great.";
    // std::vector<float> embedded1 =  executor.calculateEmbeddings(text1);

    // std::string text2 = "i am glad to hear you are fine";
    // std::vector<float> embedded2 =  executor.calculateEmbeddings(text2);

    // std::cout << "Similarity: " << executor.getSimilarity(embedded1, embedded2) << std::endl;


    std::string input;
    std::cin >> input;

    return 0;
}
