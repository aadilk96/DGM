// Deep Neural Network (based on OpenCV) training class inference
// In development by Aadil Anil Kumar in 2020

// include only once
#pragma once

#include "TrainNode.h"

namespace DirectedGraphicalModels
{
class CSamplesAccumulator;

typedef struct TrainNodeCvDNNParams
{
    // Number of layers of neurons
    word numLayers;
    // Strength of weight gradient term.
    double weightscale;
    // Strength of momentum term
    double momentumScale;
    // Max number of iterations (time/accuracy)
    int maxCount;
    // The desired accuracy or change in parameters at which the iterative alg. stops
    double epsilon;
    // Terminatino criteria type
    int term_criteria_type;
    // Max number of samples to be used in training. 0 means using all the samples
    size_t maxSamples;

    TrainNodeCvDNNParams() {}
    TrainNodeCvDNNParams(word _numLayers, double _weightScale, double _momentumScale, int _maxCount, double _epsilon, int _term_criteria_type, int _maxSamples) : numLayers(_numLayers), weightScale(_weightScale), maxCount(_maxCount), epsilon(_epsilon), term_criteria_type(_term_criteria_type), maxSamples(_maxSamples) {}
} TrainNodeCvDNNParams;

const TrainNodeCvDNNParams TRAIN_NODE_CV_DNN_PARAMS_DEFAULT = TrainNodeCvDNNParams(
    4,                                          // Num layers
    0.0001,                                     // Backpropagation Weight Scale
    0.1,                                        // Backpropagation Momentum Scale
    100,                                        // The maximum number of iterations (time / accuracy)
    0.01,                                       // The desired accuracy or change in parameters at which the iterative algorithm stops
    TermCriteria::MAX_ITER | TermCriteria::EPS, // Termination cirteria (according the the two previous parameters)
    0                                           // Maximum number of samples to be used in training. 0 means using all the samples
);

// ====================== OpenCV Deep Neural Network Train Class =====================
class CTrainNodeCvDNN : public CTrainNode
{
public:
    DllExport CTrainNodeCvDNN(byte nStates, word nFeatures, TrainNodeCvDNNParams params = TRAIN_NODE_CV_DNN_PARAMS_DEFAULT);
    DllExport CTrainNodeCvDNN(byte nStates, word nFeatures, size_t maxSamples);
    DllExport virtual ~CTrainNodeCvDNN(void);

    DllExport void reset(void);
    DllExport void save(const std::string &path, const std::string &name = std::string(), short idx = -1) const;
    DllExport void load(const std::string &path, const std::string &name = std::string(), short idx = -1);

    DllExport void addFeatureVec(const Mat &featureVector, byte gt);

    DllExport void train(bool doClean = false);

protected:
    DllExport void saveFile(FILE *pFile) const {}
    DllExport void loadFile(FILE *pFile) {}
    DllExport void calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const;

private:
    void init(TrainNodeCvDNNParams params); // This function is called by both constructors

protected:
    Ptr<ml::DNN_MLP> m_pDNN;            ///< Deep Neural Network
    CSamplesAccumulator *m_pSamplesAcc; ///< Samples Accumulator
};
} // namespace DirectedGraphicalModels
