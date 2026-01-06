#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <chrono>

using namespace nvinfer1;

// Logger simplu obligatoriu pentru API
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) { // Afisam doar erorile si avertismentele importante
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

Logger gLogger;

// Functie citire fisier binar
std::vector<char> loadEngineFile(const std::string& fileName) {
    std::ifstream file(fileName, std::ios::binary | std::ios::ate); // deschide fisierul la sfarsit ate=at end
    if (!file.good()) {// verifica daca fisierul s-a deschis corect
        std::cerr << "Eroare citire fisier: " << fileName << std::endl; // mesaj eroare
        return {};
    }
    size_t size = file.tellg();// obtine dimensiunea fisierului (cursor la final fisier aici)
    file.seekg(0, std::ios::beg);// muta cursor la inceput
    std::vector<char> buffer(size); // aloca buffer de dimensiunea fisierului
    file.read(buffer.data(), size); // citeste tot continutul in buffer
    return buffer;
}

// Constante YOLOv8
const int INPUT_W = 640;
const int INPUT_H = 640;
const int NUM_CLASSES = 80; //tipul de obiecte din COCO pe care a fost antrenat modelul
//0= persoana, 1= bicicleta, 2= masina, etc.
//COCO=Common Objects in Context

int main() {
    std::cout << "\n============================================" << std::endl;
    std::cout << "  RULARE TRASARE TensorRT pe JATSON ORIN NANO " << std::endl;
    std::cout << "==============================================\n" << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    // Pas 1: Citim fisierul
    auto t1 = std::chrono::high_resolution_clock::now();
    std::string engineFile = "yolov8n.engine";
    std::cout << "[INFO] Incarcare fisier engine: " << engineFile << std::endl;
    std::vector<char> engineData = loadEngineFile(engineFile); //apel functie citire fisier
    if (engineData.empty()) return -1;

    // Pas 2: Cream Runtime-ul
    // "createInferRuntime" returneaza un raw pointer catre IRuntime
    IRuntime* runtime = createInferRuntime(gLogger); //parametru clasa logger pt gestionare erori
    if (!runtime) {
        std::cerr << "Eroare la crearea Runtime!" << std::endl;
        return -1;
    } // rol de deserializare a engine-ului ( este o instanta a motorului de inferenta)

    // Pas 3: Deserializam Engine-ul
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size()); //metoda de deserializare din runtime
    if (!engine) {
        std::cerr << "Eroare la deserializarea Engine!" << std::endl;
        // cleanup pointeri
        delete runtime;  //delete cheama destructorul si elibereaza memoria
        return -1;
    }
    // Pas 4: Cream Contextul de Executie
    // se creaza un context de executie din engine ca un proces separat de inferenta
    // aloca resursele necesare pentru executie
    // contextul e ca un thread de executie pentru engine
    // se aloca memorie pentru tensori, se seteaza pointeri, etc
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Eroare la crearea Contextului!" << std::endl;
        delete engine;
        delete runtime;
        return -1;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMP] Incarcare engine: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n" << std::endl;




    // === 2. PREGATIRE DATE (OpenCV & CUDA) ===
    auto t3 = std::chrono::high_resolution_clock::now();
    cv::Mat img = cv::imread("cars.jpg");
    if (img.empty()) {
        std::cerr << "Imagine lipsa!" << std::endl; //cerr print mesaj eroare
        // cleanup pointeri
        delete context;
        delete engine;
        delete runtime;
        return -1;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMP] Citire imagine: " << std::chrono::duration<double, std::milli>(t4 - t3).count() << " ms\n" << std::endl;



    auto t5 = std::chrono::high_resolution_clock::now();
    cv::Mat blob; //blob=Binary Large Object
    //o matrice 4d care contine imaginea preprocesata
    //N(Number/Batch Size), C(Channels), H(Height), W(Width)\
    //1x3x640x640
    cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, cv::Size(INPUT_W, INPUT_H), cv::Scalar(0, 0, 0), true, false);
    //1.0/255.0  factor de scalare pentru normalizare la [0,1]
    //cv::size(INPUT_W, INPUT_H)=dimensiunea la care se redimensioneaza imaginea primita
    //true=converteste BGR in RGB //bgr=blue green red spatiul de culoare default in OpenCV
    //false=nu face crop imagine //nu taie din imagine
    // cv::Scalar(0,0,0)=valoarea medie scazuta din fiecare canal (aici 0 deci nu se scade nimic)

    void* buffers[2];
    // Calcul dimensiuni
    int inputSize = 1 * 3 * INPUT_H * INPUT_W * sizeof(float);//dimensiune input
    // Output: [1, 84, 8400] standard pentru yolov8
    int outputElements = 1 * (4 + NUM_CLASSES) * 8400;
    int outputSize = outputElements * sizeof(float);

    // Alocare GPU pt input si output
    // mutare date din memoria CPU in memoria GPU
    cudaMalloc(&buffers[0], inputSize);
    cudaMalloc(&buffers[1], outputSize);

    // Copiere imagine(blob) pe GPU
    cudaMemcpy(buffers[0], blob.ptr<float>(), inputSize, cudaMemcpyHostToDevice);
    //buffers[0]=adresa buffer input pe GPU
    //blob.ptr<float>()=pointer la datele din blob (imaginea preprocesata)
    //inputSize=dimensiunea datelor de copiat
    //cudaMemcpyHostToDevice=directia copierei (CPU->GPU)
    auto t6 = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMP] Preprocesare + Alocare GPU: " << std::chrono::duration<double, std::milli>(t6 - t5).count() << " ms\n" << std::endl;

    // === 3. INFERENTA ===

    // Setam pointerii catre bufferele GPU in Context
    auto t7 = std::chrono::high_resolution_clock::now();
    context->setInputTensorAddress("images", buffers[0]);
    context->setTensorAddress("output0", buffers[1]);
    // "images" si "output0" sunt numele tensorilor definite in modelul YOLOv8
    // buffers[0] este inputul, buffers[1] este outputul
    // mai ok cu SetOutputTensorAddress in loc de SetTensorAddress, dar ambele functioneaza
    // depinde de versiunea TensorRT

    // Cream stream CUDA pentru executie asincrona
    // cudaStream_t = o coada de comenzi care ruleaza asincron pe GPU
    // un to do list pentru GPU de la CPU
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Lansam executia
    context->enqueueV3(stream); //lansare stream asincrona

    // Asteptam sa termine GPU-ul
    cudaStreamSynchronize(stream); //blocheaza CPU pana cand toate operatiile din stream sunt terminate
    //altfel am avea junk in output (pentru ca e asincron)

    auto t8 = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMP] *** INFERENTA GPU ***: " << std::chrono::duration<double, std::milli>(t8 - t7).count() << " ms <<<\n" << std::endl;

    // === 4. RECUPERARE REZULTATE ===
    auto t9 = std::chrono::high_resolution_clock::now();
    std::vector<float> cpuOutput(outputElements);
    //outputElements=dimensiunea output-ului in float-uri
    cudaMemcpy(cpuOutput.data(), buffers[1], outputSize, cudaMemcpyDeviceToHost);
    //cpuOutput.data()=pointer la datele din vectorul cpuOutput unde se vor copia rezultatele
    //buffers[1]=adresa buffer output pe GPU
    //outputSize=dimensiunea datelor de copiat
    //cudaMemcpyDeviceToHost=directia copierei (GPU->CPU)
    auto t10 = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMP] Copiere rezultate GPU->CPU: " << std::chrono::duration<double, std::milli>(t10 - t9).count() << " ms\n" << std::endl;

    // === 5. POST-PROCESARE (Matematica YOLO) ===
    //prelucrarea rezultatelor brute pentru a obtine detectiile finale
    auto t11 = std::chrono::high_resolution_clock::now();
    std::vector<int> classIds; //vector de id-uri de clase detectate
    std::vector<float> confidences; //vector de scoruri de incredere
    std::vector<cv::Rect> boxes; //vector de dreptunghiuri pt boxele de delimitare

    //matricea output este [1, 84, 8400]
    //84-4 coordonate + 80 clase (x_center, y_center, width, height, class0, class1,...class79)
    //8400- numar de ancore (predictions)
    int rows = 8400;//numar de ancore (predictions)
    float x_factor = (float)img.cols / INPUT_W; //factor scalare latime
    float y_factor = (float)img.rows / INPUT_H; //factor scalare inaltime
    //poza originala 1920x1080 etc. , input model 640x640 se scalaeaza inapoi la dim originala
    float* data = cpuOutput.data(); //pointer la datele output-ului
    //anchor free pt YOLOv8
    for (int i = 0; i < rows; ++i) { //parcurgem fiecare ancorare
        float* classes_scores = data + 4 * rows + i; //pointer la scorurile claselor pentru ancorarea i
        float maxClassScore = 0.0; //initializam scor maxim clasa
        int maxClassId = -1; //initializam id clasa maxima


        // Cautam clasa cu scor maxim pentru ancora curenta
        // reprezinta obiectul detectat cu cea mai mare probabilitate
        // adica din cele 80 de clase posibile daca clasa 2 de ex are cel mai mare scor probabil obiectul apartine clasei 2
        for (int c = 0; c < NUM_CLASSES; ++c) { //parcurgem fiecare clasa
            float score = data[(4 + c) * rows + i]; //accesam scorul clasei c pentru ancorarea i
            if (score > maxClassScore) {//daca scorul curent e mai mare decat maximul gasit
                maxClassScore = score;//actualizam scorul maxim
                maxClassId = c; //actualizam id clasa maxima
            }
        }
        // ia doar clasele cu scor peste un prag mai mare decat cel setat
        if (maxClassScore > 0.30) { // Prag de confidenta acceptabil
            // Extragem coordonatele (cx, cy, w, h)
            float cx = data[0 * rows + i];
            float cy = data[1 * rows + i];
            float w = data[2 * rows + i];
            float h = data[3 * rows + i];

            // calculam coordonatele casetei de delimitare in dimensiunile originale ale imaginii (boxul de pe imagine)
            int left = int((cx - 0.5 * w) * x_factor);
            int top = int((cy - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            // stocam rezultatele
            boxes.push_back(cv::Rect(left, top, width, height)); //cv::Rect=clasa OpenCV pentru dreptunghiuri
            confidences.push_back(maxClassScore);
            classIds.push_back(maxClassId);
        }
    }
    //cand parcurg ancorele, pentru fiecare ancorare gasesc clasa cu scorul maxim
    //cand fac boxul(cu cv::Rect) pt ancorle apropiate pot avea boxuri suprapuse
    //de aceea aplic NMS (Non-Maximum Suppression) pentru a elimina suprapunerile
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.30, 0.5, indices); //deseneaza doar dreptunghiuile dupa NMS adica cele cu scorul cel mai mare si elimina suprapunerile
    //NMSBoxes(vector de boxe, vector de scoruri, prag scor, prag NMS, vector de indici rezultati)
     for (int idx : indices) {
        cv::Rect box = boxes[idx];
        cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);//deseneaza dreptunghi pe imagine (verde, grosime 2)

        // Cream label-ul cu ID clasa si scor de confidenta
        // ex: "2: 0.87" inseamna clasa 2 (masina) cu 87% confidenta
        std::string label = std::to_string(classIds[idx]) + ": " + std::to_string(confidences[idx]).substr(0, 4);
        // std::to_string(confidences[idx]).substr(0, 4) = ia primele 4 caractere din scor (ex: 0.87 in loc de 0.876543)

        // Punem textul deasupra boxului (y - 5 pixeli deasupra coltului stanga-sus)
        cv::putText(img, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        // cv::Point(box.x, box.y - 5) = pozitie text (cu 5 pixeli mai sus decat boxul)
        // cv::FONT_HERSHEY_SIMPLEX = font standard
        // 0.5 = marime font
        // cv::Scalar(0, 255, 0) = culoare verde (BGR format)
        // 2 = grosime text
    }
    auto t12 = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMP] Post-procesare (NMS + Desenare): " << std::chrono::duration<double, std::milli>(t12 - t11).count() << " ms" << std::endl;
    std::cout << "[INFO] Detectii gasite: " << indices.size() << "\n" << std::endl;

    auto t13 = std::chrono::high_resolution_clock::now();
    cv::imwrite("result_test.jpg", img); //salveaza imaginea rezultata
    std::cout << "Done." << std::endl;
    auto t14 = std::chrono::high_resolution_clock::now();
    std::cout << "[TIMP] Salvare imagine: " << std::chrono::duration<double, std::milli>(t14 - t13).count() << " ms\n" << std::endl;

    // === 6. CURATARE MANUALA ===

    // Eliberam resursele CUDA
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);

    // Eliberam obiectele TensorRT manual in ordine inversa crearii
    // In versiunile moderne C++ API, se foloseste delete
    delete context;
    delete engine;
    delete runtime;

    auto total_end = std::chrono::high_resolution_clock::now();
    std::cout << "========================================" << std::endl;
    std::cout << "[TIMP TOTAL]: " << std::chrono::duration<double, std::milli>(total_end - total_start).count() << " ms" << std::endl;
    std::cout << "========================================\n" << std::endl;

    return 0;
}
