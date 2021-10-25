const float x_dat2[142*4] = { 
     0.406802,0.726248,0.369048,0.633262,0.453019,0.628019,0.380952,0.765458,
     0.369523,0.710145,0.25    ,0.795309,0.256159,0.740741,0.166667,0.889126,
     0.257467,0.668277,0.255952,0.746269,0.292784,0.776167,0.184524,0.637527,
     0.46392 ,0.723027,0.303571,0.773987,0.259865,0.644122,0.232143,0.80597 ,
     0.434707,0.618357,0.261905,0.660981,0.440375,0.602254,0.107143,0.660981,
     0.233486,0.755233,0.178571,0.577825,0.269675,0.752013,0.196429,0.663113,
     0.243078,0.5781  ,0.315476,0.673774,0.477654,0.745572,0.35119 ,0.635394,
     0.278613,0.610306,0.267857,0.812367,0.369305,0.658615,0.232143,0.597015,
     0.303466,0.636071,0.160714,0.682303,0.436015,0.700483,0.214286,0.654584,
     0.360148,0.697262,0.327381,0.682303,0.219533,0.798712,0.220238,0.720682,
     0.327665,0.681159,0.178571,0.795309,0.477218,0.608696,0.232143,0.705757,
     0.373883,0.665056,0.238095,0.603411,0.431437,0.586151,0.35119 ,0.733476,
     0.502725,0.668277,0.303571,0.82516 ,0.393285,0.68599 ,0.244048,0.765458,
     0.423588,0.679549,0.309524,0.705757,0.257249,0.777778,0.333333,0.827292,
     0.300632,0.708535,0.232143,0.793177,0.490081,0.727858,0.327381,0.746269,
     0.286462,0.689211,0.333333,0.695096,0.443427,0.822866,0.238095,0.688699,
     0.378461,0.766506,0.255952,0.686567,0.227382,0.716586,0.369048,0.641791,
     0.413342,0.716586,0.27381 ,0.652452,0.232832,0.753623,0.238095,0.773987,
     0.175932,0.769726,0.261905,0.767591,0.404622,0.780998,0.25    ,0.795309,
     0.265315,0.663446,0.160714,0.754797,0.24068 ,0.760064,0.285714,0.671642,
     0.441901,0.623188,0.142857,0.720682,0.296926,0.658615,0.380952,0.754797,
     0.446479,0.616747,0.279762,0.863539,0.346632,0.692432,0.357143,0.769723,
     0.248092,0.640902,0.357143,0.648188,0.376063,0.62963 ,0.208333,0.710021,
     0.202311,0.729469,0.220238,0.671642,0.380859,0.671498,0.327381,0.686567,
     0.290168,0.705314,0.220238,0.571429,0.380423,0.655395,0.369048,0.776119,
     0.316111,0.624799,0.160714,0.686567,0.277523,0.513688,0.130952,0.727079,
     0.191411,0.694042,0.261905,0.697228,0.469806,0.660225,0.315476,0.799574,
     0.303248,0.740741,0.327381,0.729211,0.333333,0.665056,0.244048,0.808102,
     0.275343,0.574879,0.261905,0.624733,0.42337 ,0.690821,0.244048,0.750533,
     0.559843,0.768116,0.244048,0.560768,0.255287,0.668277,0.333333,0.810235,
     0.272727,0.68277 ,0.27381 ,0.778252,0.229998,0.740741,0.255952,0.746269,
     0.459996,0.584541,0.392857,0.61194 ,0.293656,0.648953,0.357143,0.667377,
     0.298234,0.663446,0.422619,0.607676,0.380641,0.668277,0.172619,0.654584,
     0.418138,0.718196,0.321429,0.842218,0.240244,0.674718,0.303571,0.646055,
     0.428603,0.772947,0.333333,0.752665,0.282974,0.669887,0.333333,0.690832,
     0.204273,0.677939,0.267857,0.646055,0.178984,0.62963 ,0.25    ,0.69936 ,
     0.457816,0.648953,0.261905,0.714286,0.296708,0.673108,0.434524,0.759062,
     0.245912,0.603865,0.220238,0.656716,0.396119,0.613527,0.35119 ,0.616205,
     0.332679,0.766506,0.333333,0.620469,0.211903,0.582931,0.327381,0.765458,
     0.248964,0.63124 ,0.297619,0.673774,0.456071,0.653784,0.470238,0.603411,
     0.300414,0.648953,0.345238,0.633262,0.393067,0.697262,0.238095,0.695096,
     0.370831,0.586151,0.220238,0.690832,0.315675,0.653784,0.315476,0.607676,
     0.415958,0.732689,0.303571,0.769723,0.387181,0.618357,0.315476,0.667377,
     0.341182,0.68438 ,0.291667,0.584222,0.223458,0.745572,0.363095,0.748401,
     0.298452,0.653784,0.363095,0.609808,0.290822,0.487923,0.279762,0.597015,
     0.30085 ,0.594203,0.279762,0.609808,0.413996,0.637681,0.357143,0.686567,
     0.494877,0.679549,0.363095,0.584222,0.266623,0.645733,0.375   ,0.622601,
     0.143013,0.653784,0.315476,0.688699,0.284064,0.561997,0.321429,0.633262,
     0.539132,0.571659,0.363095,0.635394,0.437541,0.466989,0.309524,0.771855,
     0.472858,0.62963 ,0.327381,0.58209 ,0.405276,0.703704,0.470238,0.609808,
     0.286462,0.566828,0.458333,0.682303,0.393939,0.574879,0.380952,0.573561,
     0.223022,0.610306,0.428571,0.579957,0.427077,0.613527,0.291667,0.78678 ,
     0.375845,0.52496 ,0.27381 ,0.665245,0.392631,0.603865,0.35119 ,0.601279,
     0.52518 ,0.626409,0.297619,0.652452,0.324177,0.73752 ,0.452381,0.603411,
     0.335949,0.602254,0.434524,0.624733,0.41356 ,0.681159,0.416667,0.707889,
     0.290822,0.586151,0.458333,0.605544,0.336167,0.537842,0.220238,0.684435,
     0.431873,0.492754,0.363095,0.652452,0.378897,0.714976,0.434524,0.637527,
     0.518858,0.689211,0.345238,0.569296,0.202093,0.732689,0.345238,0.620469,
     0.408328,0.763285,0.39881 ,0.539446,0.318945,0.708535,0.416667,0.558635,
     0.561587,0.63124 ,0.416667,0.590618,0.495749,0.613527,0.446429,0.565032,
     0.490953,0.619968,0.35119 ,0.556503,0.493569,0.57649 ,0.446429,0.47548 ,
     0.340528,0.449275,0.333333,0.620469,0.350992,0.388084,0.416667,0.577825,
     0.314585,0.417069,0.464286,0.614073,0.252889,0.623188,0.380952,0.567164,
     0.34576 ,0.637681,0.464286,0.565032,0.497711,0.465378,0.458333,0.697228,
     0.442337,0.413849,0.452381,0.710021,0.323959,0.452496,0.363095,0.597015,
     0.321343,0.619968,0.505952,0.588486,0.545455,0.533011,0.422619,0.628998,
     0.470896,0.639291,0.428571,0.541578,0.274253,0.539452,0.535714,0.603411,
     0.316329,0.521739,0.291667,0.490405,0.298234,0.697262,0.345238,0.607676,
     0.515588,0.570048,0.619048,0.626866,0.409418,0.492754,0.327381,0.614073,
     0.463048,0.547504,0.488095,0.498934,0.432745,0.37037 ,0.392857,0.550107,
     0.413778,0.57971 ,0.380952,0.505331,0.528668,0.52496 ,0.440476,0.373134  };
