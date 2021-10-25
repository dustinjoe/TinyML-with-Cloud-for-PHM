#include <Arduino.h>
/* ESP32 Dependencies */
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <SPIFFS.h>



/* Your WiFi Credentials */
const char* ssid = ""; // SSID
const char* password = ""; // Password



// IP Address details
IPAddress local_ip(192, 168, 1, 1);
IPAddress gateway(192, 168, 1, 1);
IPAddress subnet(255, 255, 255, 0);



/* Start Webserver */
AsyncWebServer server(80);




#include <EloquentTinyML.h>
#include "pred2_16_seq10.h"
#include "x_dat.h"

#define NUMBER_OF_INPUTS 40
#define NUMBER_OF_OUTPUTS 1
// in future projects you may need to tweak this value: it's a trial and error process
#define TENSOR_ARENA_SIZE 8*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml_pred;

float sensor_x_t = x_dat[3];
//String sensor_x_str;
float rul_pred =0.0; 
//String rul_pred_str;

int timestep = 0;
const int timelen = 142;


void connectToWiFi() {
    Serial.println("Connecting to WiFi network");
    WiFi.disconnect(true);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
    }
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
}

void setWifiAP() {
      // Create SoftAP
    WiFi.softAP(ssid, password);
    WiFi.softAPConfig(local_ip, gateway, subnet);
  
  
    Serial.print("Connect to My access point: ");
    Serial.println(ssid);
}


void setup() {
    Serial.begin(115200);
    connectToWiFi();

    
    ml_pred.begin(pred2_16_seq10);


    // Initialize SPIFFS
    if(!SPIFFS.begin()){
      Serial.println("An Error has occurred while mounting SPIFFS");
      return;
    }


    
    // Route for root / web page
    server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
      request->send(SPIFFS, "/index.html");
    });

    server.on("/inputsensor", HTTP_GET, [sensor_x_t](AsyncWebServerRequest *request){
      request->send(200, "text/plain", String(sensor_x_t));
    });
    
    server.on("/rulpredvalue", HTTP_GET, [rul_pred](AsyncWebServerRequest *request){
      request->send(200, "text/plain", String(rul_pred));
    });

    /* Start AsyncWebServer */
    server.begin();

  

}

float mlPredRUL(float x_in[]) {
  
    float rulpred[1] = {0};
  
    //uint32_t start = micros();  
  
    ml_pred.predict(x_in,rulpred);
  
    //uint32_t timeit = micros() - start;
  
    //Serial.print("It took ");
    //Serial.print(timeit);
    //Serial.println(" micros to run inference");
    //Serial.print("Predicted RUL is: ");
    //Serial.println(rulpred[0]);
  
  
    //delay(200);
  
  
    return rulpred[0];
}

void loop() {
    // pick up a  x and predict its sine
    timestep = timestep%timelen;

    // simulate an incoming data stream as 142 time steps, 4 sensor features each step
    // each model input contains sensor values from past 10 steps
    int x_i;
    float x_in_t[40] = {0};


    // get t's step's 4 sensor values
    // 4 sensors in this group are related to HPC: 
    // 1.total tmp HPC outlet; 2.total pres HPC outlet; 3.static pres HPC outlet(Ps30); 4.ratio of fuel to Ps30
    for( x_i = 0; x_i < 40; x_i = x_i + 1 ){
      x_in_t[x_i] = x_dat[timestep*4+x_i];
    }
    sensor_x_t = x_dat[timestep*4+1]; //visualize the 2nd sensor in this group

    uint32_t start = micros();  
    rul_pred = mlPredRUL(x_in_t);
    uint32_t timeit = micros() - start;
  
    Serial.print("It took ");
    Serial.print(timeit);
    Serial.println(" micros to run inference");

    Serial.print("Predicted RUL is: ");
    Serial.println(rul_pred);


    server.on("/inputsensor", HTTP_GET, [sensor_x_t](AsyncWebServerRequest *request){
      request->send(200, "text/plain", String(sensor_x_t));
    });
    
    server.on("/rulpredvalue", HTTP_GET, [rul_pred](AsyncWebServerRequest *request){
      request->send(200, "text/plain", String(rul_pred));
    });

    timestep = timestep+1;
    delay(650);     

    
}
