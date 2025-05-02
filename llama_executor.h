// LlamaExecutor.h
#ifndef LLAMA_EXECUTOR_H
#define LLAMA_EXECUTOR_H

#include "llama.h"
#include "common.h"
#include <string>
#include <vector>
#include <iostream>
#include <functional>


class LlamaExecutor {
public:
    LlamaExecutor();
    ~LlamaExecutor();

    void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id);
    void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd, int embd_norm);

    void setOptions(const std::string& model_path, int ngl = 99, int n_ctx = 2048, float minP = 0.05f, float temp = 0.8f, int topK = 50, float topP = 0.9, bool embedding = false);
    void setCallBackFunction(std::function<void(const std::string&)> func);
    std::string generateResponse(const std::string& prompt);
    void chat(const std::string &userInput);
    std::vector<float> calculateEmbeddings(const std::string& text);
    float getSimilarity(const std::vector<float>& Emb1, const std::vector<float>& Emb2){ return common_embd_similarity_cos(&Emb1[0], &Emb2[0], Emb1.size()); }

private:
    void initializeModel();
    void freeResources();

    std::string _modelPath;
    int _ngl;
    int _nCtx;
    float _minP;
    float _temp;
    int _topK;
    float _topP;
    bool _embedding;

    llama_model* _model;
    llama_context* _ctx;
    llama_sampler* _smpl;
    const llama_vocab* _vocab;
    std::vector<llama_chat_message> _messages;
    std::function<void(const std::string&)> _responseCallbackFunction;

};

#endif // LLAMA_EXECUTOR_H
