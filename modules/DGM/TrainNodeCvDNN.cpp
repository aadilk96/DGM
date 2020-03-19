#include "TrainNodeCvDNN.h"
#include "SamplesAccumulator.h"

namespace DirectGraphicalModels
{
// Constructor
CTrainNodeCvDNN::CTrainNodeCvDNN(byte nStates, word nFeatures, TrainNodeCvDNNParams params) : CBaseRandomModel(nStates), CTrainNode(nStates, nFeatures)
{
    init(params);
}

// Constructor
CTrainNodeCvDNN::CTrainNodeCvDNN(byte nStates, word nFeatures, size_t maxSamples) : CBaseRandomModel(nStates), CTrainNode(nStates, nFeatures)
{
    TrainNodeCvDNNParams params = TRAIN_NODE_CV_DNN_PARAMS_DEFAULT;
    params.maxSamples = maxSamples;
    init(params);
}

void CTrainNodeCvDNN::init(TrainNodeCvDNNParams params)
{
    m_pSamplesAcc = new CSamplesAccumulator(m_nStates, params.maxSamples);

    if (params.numLayers < 2)
        params.numLayers = 2;
    std::vector<int> vLayers(params.numLayers);
    vLayers[0] = getNumFeatures();
    for (int i = 1; i < params.numLayers - 1; i++)
        vLayers[i] = m_nStates * 1 << (params.numLayers - i);
    vLayers[params.numLayers - 1] = m_nStates;

    m_pDNN = ml::DNN_MLP::create();
    m_pDNN->setLayerSizes(vLayers);
    m_pDNN->setActivationFunction(ml::DNN_MLP::SIGMOID_SYM, 0.0, 0.0);
    m_pDNN->setTermCriteria(TermCriteria(params.term_criteria_type, params.maxCount, params.epsilon));
    m_pDNN->setTrainMethod(ml::DNN_MLP::BACKPROP, params.weightScale, params.momentumScale);
}

// Destructor
CTrainNodeCvDNN::~CTrainNodeCvDNN(void)
{
    delete m_pSamplesAcc;
}

void CTrainNodeCvDNN::reset(void)
{
    m_pSamplesAcc->reset();
    m_pDNN->clear();
}

void CTrainNodeCvDNN::save(const std::string &path, const std::string &name, short idx) const
{
    std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvDNN" : name, idx);
    m_pDNN->save(fileName.c_str());
}

void CTrainNodeCvDNN::load(const std::string &path, const std::string &name, short idx)
{
    std::string fileName = generateFileName(path, name.empty() ? "TrainNodeCvDNN" : name, idx);
    m_pDNN = Algorithm::load<ml::DNN_MLP>(fileName.c_str());
}

void CTrainNodeCvDNN::addFeatureVec(const Mat &featureVector, byte gt)
{
    m_pSamplesAcc->addSample(featureVector, gt);
}

void CTrainNodeCvDNN::train(bool doClean)
{
#ifdef DEBUG_PRINT_INFO
    printf("\n");
#endif
    // Filling the <samples> and <classes>
    Mat samples, classes;
    for (byte s = 0; s < m_nStates; s++)
    { // states
        int nSamples = m_pSamplesAcc->getNumSamples(s);
#ifdef DEBUG_PRINT_INFO
        printf("State[%d] - %d of %d samples\n", s, nSamples, m_pSamplesAcc->getNumInputSamples(s));
#endif
        samples.push_back(m_pSamplesAcc->getSamplesContainer(s));
        Mat classes_s(nSamples, m_nStates, CV_32FC1, Scalar(0.0f)); // CV_32FC1 defines the depth and channel of image
        classes_s.col(s).setTo(1.0f);
        classes.push_back(classes_s);
        if (doClean)
            m_pSamplesAcc->release(s); // free memory
    } // s
    samples.convertTo(samples, CV_32FC1);

    // Training
    try
    {
        m_pDNN->train(samples, ml::ROW_SAMPLE, classes);
    }
    catch (std::exception &e)
    {
        printf("EXCEPTION: %s\n", e.what());
        getchar();
        exit(-1);
    }
}

void CTrainNodeCvDNN::calculateNodePotentials(const Mat &featureVector, Mat &potential, Mat &mask) const
{
    Mat fv;
    featureVector.convertTo(fv, CV_32FC1); 

    m_pDNN->predict(fv.t(), potential);
    for (float &pot : static_cast<Mat_<float>>(potential))
        if (pot < 0)
            pot = 0;
    potential = potential.t();
}
} // namespace DirectGraphicalModels