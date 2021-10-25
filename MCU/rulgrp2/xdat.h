float x_test[200] = { 0.4 ,0.49,0.49,0.59,
                          0.47,0.55,0.43,0.56,
                          0.34,0.54,0.51,0.56,
                          0.53,0.67,0.49,0.59,
                          0.47,0.59,0.51,0.49,
                          0.34,0.54,0.36,0.53,
                          0.47,0.56,0.51,0.57, 
                          0.45,0.53,0.44,0.54,
                          0.44,0.49,0.43,0.49,
                          0.48,0.66,0.38,0.54,
                          0.37,0.6, 0.54,0.52,
                          0.56,0.5 ,0.46,0.56,
                          0.45,0.48,0.45,0.48,
                          0.31,0.51,0.42,0.57,
                          0.45,0.55,0.34,0.57,
                          0.49,0.52,0.33,0.43,
                          0.49,0.49,0.38,0.52,
                          0.38,0.48,0.47,0.54,
                          0.53,0.51,0.45,0.49,
                          0.39,0.5 ,0.46,0.57,
                          0.39,0.52,0.36,0.55,
                          0.55,0.45,0.4 ,0.59,
                          0.52,0.59,0.39,0.58,
                          0.63,0.46,0.42,0.48,
                          0.35,0.51,0.43,0.43,
                          0.4 ,0.53,0.45,0.5 ,
                          0.4 ,0.62,0.4 ,0.55,
                          0.45,0.53,0.55,0.6 ,
                          0.33,0.44,0.44,0.53,
                          0.41,0.48,0.54,0.51,
                          0.4 ,0.37,0.43,0.6 ,
                          0.49,0.52, 0.44,0.59,
                          0.49,0.53,0.45,0.57,
                          0.25,0.52,0.44,0.57,
                          0.41,0.48,0.49,0.48,
                          0.47,0.47,0.48,0.42,
                          0.53,0.62,0.43,0.42,
                          0.46,0.48,0.58,0.65,
                          0.47,0.57,0.49,0.51,
                          0.46,0.61,0.48,0.48,
                          0.46,0.48,0.57,0.45,
                          0.55,0.53,0.52,0.43,
                          0.41,0.53,0.36,0.48,
                          0.62,0.4 ,0.61,0.5 ,
                          0.53,0.56,0.47,0.46,
                          0.58,0.48,0.54,0.55,
                          0.55,0.46,0.49,0.39,
                          0.42,0.39,0.45,0.46,
                          0.38,0.52,0.46,0.31,
                          0.41,0.44,0.61,0.46  };


float x_test[40] = {      0.46,0.48,0.57,0.45,
                          0.55,0.53,0.52,0.43,
                          0.41,0.53,0.36,0.48,
                          0.62,0.4 ,0.61,0.5 ,
                          0.53,0.56,0.47,0.46,
                          0.58,0.48,0.54,0.55,
                          0.55,0.46,0.49,0.39,
                          0.42,0.39,0.45,0.46,
                          0.38,0.52,0.46,0.31,
                          0.41,0.44,0.61,0.46  };




#include <EloquentTinyML.h>
#include "enc2_16_seq10.h"

#define NUMBER_OF_INPUTS 200
#define NUMBER_OF_OUTPUTS 32
// in future projects you may need to tweak this value: it's a trial and error process
#define TENSOR_ARENA_SIZE 80*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml_enc;

#include "lat2_16_seq10.h"

#define NUMBER_OF_INPUTS_SINE 32
#define NUMBER_OF_OUTPUTS_SINE 1
// in future projects you may need to tweek this value: it's a trial and error process
#define TENSOR_ARENA_SIZE_SINE 8*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS_SINE, NUMBER_OF_OUTPUTS_SINE, TENSOR_ARENA_SIZE_SINE> ml_lat;



void setup() {
    Serial.begin(115200);
    ml_enc.begin(enc2_16_seq10);
    ml_lat.begin(lat2_16_seq10);
}

void loop() {
    // pick up a random x and predict its sine
    float x_test[40] = {  0.46,0.48,0.57,0.45,
                          0.55,0.53,0.52,0.43,
                          0.41,0.53,0.36,0.48,
                          0.62,0.4 ,0.61,0.5 ,
                          0.53,0.56,0.47,0.46,
                          0.58,0.48,0.54,0.55,
                          0.55,0.46,0.49,0.39,
                          0.42,0.39,0.45,0.46,
                          0.38,0.52,0.46,0.31,
                          0.41,0.44,0.61,0.46  };
    float y_pred[1] = {0};
    float y_test = 69.0;
    float enc_out[16] = {0};

    uint32_t start = micros();
    
    ml_enc.predict(x_test, enc_out);
    Serial.print("Latent Representation 16bits: ");
    for (int i = 0; i < 16; i++) {
        Serial.print(enc_out[i]);
        Serial.print(i == 16 ? '\n' : ',');
    }

    delay(1000);
    
    ml_lat.predict(enc_out,y_pred);

    uint32_t timeit = micros() - start;

    Serial.print("It took ");
    Serial.print(timeit);
    Serial.println(" micros to run inference");

    Serial.print("True RUL is: ");
    Serial.println(y_test);


    Serial.print("Predicted RUL is: ");
    Serial.println(y_pred[0]);


    delay(1000);





}
