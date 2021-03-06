
// if having troubles with min/max, uncomment the following
// #undef min    
// #undef max
#ifdef __has_attribute
#define HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define HAVE_ATTRIBUTE(x) 0
#endif
#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))
#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(4)))
#else
#define DATA_ALIGN_ATTRIBUTE
#endif
const unsigned char lat2_32_seq10[] DATA_ALIGN_ATTRIBUTE = {
	0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x12, 0x00, 
	0x1c, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00, 
	0x00, 0x00, 0x18, 0x00, 0x12, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 
	0x14, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x94, 0x00, 0x00, 0x00, 
	0x18, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 
	0xd4, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xa0, 0x00, 0x00, 0x00, 
	0x09, 0x00, 0x00, 0x00, 0x44, 0x0c, 0x00, 0x00, 0x40, 0x0c, 0x00, 0x00, 
	0x5c, 0x0b, 0x00, 0x00, 0xf4, 0x0a, 0x00, 0x00, 0xa4, 0x02, 0x00, 0x00, 
	0x14, 0x02, 0x00, 0x00, 0x2c, 0x0c, 0x00, 0x00, 0x28, 0x0c, 0x00, 0x00, 
	0x38, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 
	0x08, 0x00, 0x0c, 0x00, 0x04, 0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 
	0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 
	0x6d, 0x69, 0x6e, 0x5f, 0x72, 0x75, 0x6e, 0x74, 0x69, 0x6d, 0x65, 0x5f, 
	0x76, 0x65, 0x72, 0x73, 0x69, 0x6f, 0x6e, 0x00, 0xfa, 0xf4, 0xff, 0xff, 
	0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x31, 0x2e, 0x35, 0x2e, 
	0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
	0x0f, 0x00, 0x00, 0x00, 0x4d, 0x4c, 0x49, 0x52, 0x20, 0x43, 0x6f, 0x6e, 
	0x76, 0x65, 0x72, 0x74, 0x65, 0x64, 0x2e, 0x00, 0x00, 0x00, 0x0e, 0x00, 
	0x18, 0x00, 0x04, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x14, 0x00, 
	0x0e, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00, 
	0x34, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 
	0x07, 0x00, 0x00, 0x00, 0x5c, 0x0b, 0x00, 0x00, 0xf8, 0x0a, 0x00, 0x00, 
	0x54, 0x0a, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0xb0, 0x01, 0x00, 0x00, 
	0x08, 0x01, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 
	0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 
	0x02, 0x00, 0x00, 0x00, 0x98, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 
	0x04, 0x00, 0x00, 0x00, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00, 0x0e, 0x00, 
	0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x07, 0x00, 0x10, 0x00, 
	0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x1c, 0x00, 0x00, 0x00, 
	0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x86, 0xff, 0xff, 0xff, 
	0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 
	0x03, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 
	0x02, 0x00, 0x00, 0x00, 0x36, 0xf5, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 
	0x07, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 
	0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 
	0x08, 0x00, 0x00, 0x00, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74, 0x79, 
	0x00, 0x00, 0x00, 0x00, 0x70, 0xf5, 0xff, 0xff, 0x00, 0x00, 0x0e, 0x00, 
	0x16, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x07, 0x00, 0x10, 0x00, 
	0x0e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x24, 0x00, 0x00, 0x00, 
	0x18, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00, 
	0x08, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 
	0x01, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 
	0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 
	0x00, 0x00, 0x0a, 0x00, 0x0c, 0x00, 0x07, 0x00, 0x00, 0x00, 0x08, 0x00, 
	0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x01, 0x00, 0x00, 0x00, 
	0xce, 0xf5, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 
	0x14, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 
	0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 
	0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x5f, 0x35, 0x2f, 0x64, 0x65, 0x6e, 0x73, 
	0x65, 0x5f, 0x36, 0x2f, 0x52, 0x65, 0x6c, 0x75, 0x3b, 0x6d, 0x6f, 0x64, 
	0x65, 0x6c, 0x5f, 0x35, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x36, 
	0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x00, 0x00, 0x00, 0x00, 
	0xe4, 0xf5, 0xff, 0xff, 0xca, 0xf6, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 
	0x40, 0x00, 0x00, 0x00, 0xa0, 0x7c, 0x1f, 0x3f, 0xe8, 0x68, 0x61, 0xbe, 
	0x80, 0x94, 0x46, 0xbf, 0x7d, 0xfe, 0xe8, 0x3e, 0x15, 0xd7, 0x8d, 0x3d, 
	0xcf, 0x7e, 0xaa, 0x3e, 0xf0, 0x3c, 0x28, 0xbe, 0x37, 0xf1, 0x82, 0xbe, 
	0x5b, 0x1e, 0x01, 0x3f, 0x3a, 0x29, 0x2b, 0x3f, 0xc3, 0xd3, 0x35, 0x3f, 
	0x20, 0x9d, 0xff, 0x3e, 0xb9, 0x0c, 0x01, 0xbf, 0x7f, 0xf9, 0xf2, 0xbc, 
	0x04, 0x84, 0x23, 0xbe, 0xa4, 0x76, 0xd4, 0xbe, 0x72, 0xf6, 0xff, 0xff, 
	0x10, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 
	0x2c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 
	0x10, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x6d, 0x6f, 0x64, 0x65, 
	0x6c, 0x5f, 0x35, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x37, 0x2f, 
	0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00, 0x00, 0x70, 0xf6, 0xff, 0xff, 
	0x56, 0xf7, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 
	0x64, 0x34, 0xa3, 0x3e, 0xa5, 0x26, 0xed, 0xbd, 0x0e, 0xc3, 0xc9, 0x3e, 
	0xae, 0x8e, 0x22, 0xbe, 0x35, 0x90, 0xb1, 0x3d, 0x81, 0x91, 0x55, 0x3e, 
	0xee, 0x6a, 0x10, 0x3f, 0x4b, 0x21, 0xce, 0x3e, 0x60, 0xbc, 0x51, 0x3f, 
	0xff, 0xc3, 0x17, 0x3f, 0x5f, 0xd3, 0x78, 0x3e, 0x02, 0x1c, 0x4e, 0xbe, 
	0x19, 0x46, 0x46, 0x3f, 0x21, 0x03, 0x20, 0x3f, 0xf4, 0x98, 0xf4, 0x3e, 
	0xae, 0x81, 0x0d, 0x3e, 0x13, 0xd8, 0x07, 0x3f, 0x16, 0x61, 0x06, 0x3e, 
	0x67, 0x0a, 0x03, 0x3f, 0x37, 0x96, 0xe3, 0x3e, 0x59, 0x05, 0x7c, 0x3f, 
	0xdb, 0xb4, 0xb3, 0xbd, 0x85, 0x28, 0x0a, 0xbe, 0x70, 0x6e, 0x3a, 0x3f, 
	0xc9, 0x52, 0x4b, 0x3e, 0x1a, 0x7a, 0xa4, 0x3c, 0xb6, 0x49, 0xcb, 0x3e, 
	0x98, 0xa7, 0xd5, 0x3e, 0x3c, 0xfd, 0xf2, 0x3d, 0x2c, 0x2a, 0x42, 0x3f, 
	0x3d, 0x3e, 0xa8, 0x3e, 0x56, 0x1c, 0x2e, 0x3f, 0x08, 0x8a, 0x12, 0xbe, 
	0x6e, 0x25, 0xcc, 0xbe, 0xc9, 0x7b, 0x8d, 0xbe, 0x7e, 0x73, 0x1f, 0x3e, 
	0x23, 0x71, 0xad, 0xbe, 0x7d, 0xc4, 0x48, 0xbc, 0xc2, 0x0b, 0xb3, 0xbe, 
	0x1f, 0x8a, 0xab, 0xbe, 0xaf, 0xa2, 0xb3, 0x3d, 0x2d, 0x80, 0xaa, 0xba, 
	0x82, 0xc8, 0x70, 0xbe, 0x52, 0x90, 0x44, 0x3e, 0x41, 0xaf, 0x47, 0x3d, 
	0xf9, 0x76, 0xcc, 0x3e, 0xa9, 0xb5, 0x50, 0x3e, 0x59, 0x95, 0x2f, 0xbe, 
	0x1b, 0xcf, 0x4a, 0xbd, 0xd4, 0x5a, 0xad, 0xbd, 0x9d, 0xf3, 0x9b, 0x3e, 
	0x4b, 0xec, 0x00, 0xbe, 0x4c, 0xd4, 0x8b, 0x3d, 0xc0, 0x27, 0xb5, 0x3c, 
	0x57, 0x15, 0x91, 0x3c, 0x87, 0x5a, 0x51, 0x3d, 0x8c, 0x8c, 0xca, 0xbd, 
	0x90, 0xa8, 0x6e, 0x3e, 0x5c, 0xa2, 0x46, 0xbe, 0x9d, 0xa5, 0x63, 0xbe, 
	0x46, 0xe1, 0x85, 0x3e, 0xd9, 0xa8, 0xd7, 0xbe, 0x0f, 0x4b, 0x77, 0xbe, 
	0xcb, 0xc5, 0x18, 0x3d, 0xe8, 0x46, 0x32, 0xbf, 0x60, 0xe3, 0x85, 0x3d, 
	0xfb, 0xb2, 0x2d, 0xbe, 0xd4, 0x8d, 0x2c, 0xbc, 0x2c, 0xba, 0x1f, 0x3e, 
	0x55, 0xb5, 0x1c, 0xbe, 0x2e, 0x5b, 0x35, 0x3e, 0xbf, 0x44, 0xd1, 0x3d, 
	0x4b, 0x0b, 0xbc, 0xbe, 0x02, 0x3d, 0x2a, 0xbd, 0xe2, 0x24, 0x2d, 0x3d, 
	0x31, 0x46, 0x0b, 0x3f, 0x25, 0x70, 0xf8, 0xbe, 0xc5, 0x2a, 0x41, 0xbe, 
	0xc5, 0x2e, 0x3c, 0xbe, 0xc4, 0x24, 0xfc, 0xbe, 0x2e, 0x3b, 0x81, 0xbd, 
	0xa0, 0x3f, 0xd2, 0x3e, 0xaa, 0x35, 0xe0, 0x3e, 0x93, 0x7e, 0x98, 0x3d, 
	0xb0, 0xf2, 0xa2, 0xbe, 0x60, 0x8e, 0xc1, 0xbd, 0x70, 0x95, 0xbc, 0x3e, 
	0x48, 0x79, 0x56, 0xbf, 0xf1, 0x4a, 0x9b, 0xbd, 0x32, 0xb5, 0xa3, 0x3e, 
	0xac, 0x3b, 0x07, 0xbf, 0x5b, 0x5b, 0xa1, 0x3e, 0xa8, 0x8e, 0x1f, 0xbe, 
	0x0c, 0x31, 0xdf, 0xbd, 0x05, 0xf7, 0x2c, 0x3d, 0xf8, 0xb6, 0x32, 0xbf, 
	0x49, 0x62, 0x57, 0x3f, 0xfa, 0x95, 0xb4, 0xbd, 0xc3, 0xfc, 0xb7, 0x3e, 
	0x59, 0x65, 0xa6, 0xbd, 0xaa, 0x3d, 0x81, 0x3e, 0xd3, 0x65, 0xe6, 0x3e, 
	0xe0, 0x77, 0x9f, 0x3e, 0xce, 0x48, 0x9e, 0x3c, 0x19, 0x8f, 0x6b, 0x3e, 
	0x17, 0x6f, 0xb3, 0x3e, 0x31, 0xac, 0x39, 0x3e, 0xf8, 0x58, 0xce, 0xbc, 
	0xf4, 0xce, 0x18, 0x3f, 0x33, 0xb6, 0x83, 0x3e, 0x2f, 0x52, 0x22, 0x3f, 
	0xc7, 0xb4, 0xf8, 0x3e, 0x81, 0xd5, 0x50, 0x3e, 0x77, 0x7c, 0x05, 0x3f, 
	0x3a, 0x10, 0x0b, 0xbd, 0x80, 0xcb, 0xf7, 0x3e, 0xd3, 0x8d, 0x75, 0x3f, 
	0x39, 0x83, 0xa6, 0xbe, 0xdf, 0x56, 0x05, 0x3f, 0xf3, 0xb5, 0x3b, 0x3f, 
	0x99, 0x16, 0x3b, 0x3e, 0x5b, 0x57, 0xb6, 0x3d, 0xf1, 0x79, 0x25, 0x3f, 
	0x09, 0x40, 0x99, 0x3d, 0x42, 0xd9, 0x81, 0xbd, 0x99, 0xed, 0x4e, 0x3f, 
	0x2c, 0x6a, 0x14, 0xbe, 0x74, 0x92, 0x93, 0x3f, 0x77, 0xb5, 0x45, 0x3e, 
	0xed, 0x97, 0x09, 0xbe, 0xb4, 0xc6, 0x4d, 0xbe, 0x00, 0x00, 0x9b, 0xbd, 
	0x7a, 0x16, 0x23, 0xbe, 0x20, 0xc8, 0x8c, 0xbd, 0xf0, 0x15, 0x3b, 0x3d, 
	0xd8, 0xd3, 0xae, 0x3d, 0x30, 0xe4, 0xab, 0x3d, 0x3e, 0x5b, 0x66, 0x3e, 
	0xf5, 0xac, 0x83, 0xbc, 0xff, 0x77, 0x45, 0xbe, 0xd5, 0x1a, 0xb3, 0x3d, 
	0x05, 0x07, 0x93, 0x3e, 0xa6, 0x22, 0x28, 0x3e, 0xc0, 0xba, 0xf3, 0x3d, 
	0x90, 0x4a, 0x23, 0xbe, 0x04, 0x0e, 0xb2, 0xbe, 0x97, 0x82, 0x95, 0xbe, 
	0x5a, 0xc3, 0x2f, 0x3e, 0xab, 0x03, 0x84, 0xbe, 0x86, 0xe9, 0x56, 0x3e, 
	0xb6, 0x5a, 0x6a, 0xbe, 0x3b, 0xe8, 0xc5, 0x3d, 0xf4, 0xe6, 0xc7, 0xbd, 
	0x4e, 0x06, 0xd2, 0xbc, 0x80, 0x69, 0xa3, 0x3c, 0x91, 0x44, 0xa4, 0xbe, 
	0x4b, 0xe9, 0xa1, 0x3e, 0x07, 0x9c, 0x07, 0xbe, 0x27, 0x25, 0x0e, 0xbe, 
	0xc0, 0xda, 0x21, 0x3c, 0xce, 0xfc, 0x06, 0x3f, 0x05, 0x57, 0x7e, 0x3e, 
	0x2c, 0x2b, 0xc7, 0x3d, 0xc9, 0x22, 0x13, 0xbe, 0x27, 0xbd, 0x52, 0x3e, 
	0x61, 0x84, 0x04, 0x3f, 0x75, 0x4e, 0xce, 0x3e, 0x05, 0x64, 0x77, 0x3e, 
	0xc6, 0x8d, 0xff, 0x3e, 0xb0, 0x3b, 0xef, 0x3e, 0x17, 0x2f, 0x71, 0x3c, 
	0xc0, 0xb2, 0xad, 0xbe, 0x10, 0x6f, 0xdf, 0x3e, 0x36, 0x84, 0x26, 0x3f, 
	0xbf, 0x39, 0xb3, 0x3e, 0x7b, 0x26, 0x87, 0x3d, 0xd0, 0xe6, 0xa4, 0xbd, 
	0x44, 0xa3, 0x0c, 0x3f, 0x79, 0x30, 0x09, 0x3f, 0x03, 0x8f, 0x01, 0x3e, 
	0xd6, 0x31, 0x44, 0x3f, 0x27, 0x88, 0x84, 0xb8, 0x2d, 0xc6, 0xa4, 0x3e, 
	0xb9, 0xc2, 0x2b, 0x3f, 0xc0, 0x90, 0xbc, 0x3e, 0x65, 0x2c, 0x07, 0x3f, 
	0x9c, 0xdd, 0x29, 0x3f, 0x28, 0xe2, 0x5d, 0xbc, 0xdf, 0x59, 0xe2, 0x3e, 
	0x2a, 0x42, 0xe6, 0x3e, 0x01, 0x3e, 0x41, 0xbd, 0x1f, 0xc3, 0x84, 0x3f, 
	0x3f, 0xd2, 0x24, 0x3e, 0xb8, 0x6d, 0xbe, 0x3c, 0xfc, 0x02, 0x93, 0xbd, 
	0x61, 0x65, 0x04, 0xbe, 0x2a, 0x2a, 0xd8, 0xbd, 0xbe, 0xd5, 0x80, 0xbd, 
	0xfe, 0xb3, 0x00, 0xbe, 0xc7, 0xf6, 0xd2, 0xbd, 0xad, 0xe1, 0x6c, 0x3e, 
	0x8a, 0x55, 0xe7, 0xbd, 0x2b, 0x80, 0x03, 0x3e, 0xba, 0x4e, 0x45, 0x3e, 
	0xae, 0xe3, 0x26, 0xbe, 0xec, 0xf5, 0x2e, 0x3d, 0xf8, 0x20, 0x8c, 0x3e, 
	0xc1, 0xd7, 0xc1, 0xbe, 0x48, 0x81, 0x5b, 0x3e, 0x5a, 0xe2, 0x7a, 0x3e, 
	0x30, 0x4e, 0xc6, 0xbe, 0x0b, 0xe3, 0x6e, 0xbe, 0x5d, 0x63, 0xf6, 0xbd, 
	0xd4, 0x6a, 0x26, 0xbe, 0x9e, 0x26, 0x36, 0x3e, 0xe7, 0x38, 0x8c, 0xbe, 
	0xa5, 0x5e, 0x5c, 0x3e, 0xf7, 0xfb, 0x26, 0xbd, 0xa5, 0xd0, 0xb4, 0xbe, 
	0x8c, 0x1d, 0x94, 0xbe, 0x5b, 0x79, 0x55, 0x3e, 0x74, 0xa7, 0xa6, 0xbe, 
	0x3d, 0x46, 0x70, 0xbe, 0x8f, 0xb5, 0x45, 0x3c, 0xe5, 0xea, 0x2c, 0x3e, 
	0x85, 0x37, 0x8e, 0x3e, 0x99, 0xaa, 0x6d, 0xbe, 0x47, 0xdf, 0x1e, 0xbe, 
	0xc8, 0x4e, 0x3c, 0x3e, 0x3a, 0x93, 0xc0, 0xbe, 0x62, 0x2c, 0x37, 0x3e, 
	0x81, 0xa9, 0x5b, 0xbe, 0x0a, 0xe2, 0x3c, 0x3e, 0x29, 0x56, 0x9e, 0xbe, 
	0x70, 0x62, 0x4a, 0x3e, 0x5d, 0x27, 0xb7, 0xbe, 0xc8, 0x91, 0x22, 0x3d, 
	0x25, 0x0f, 0x48, 0x3e, 0x90, 0x07, 0xa8, 0xbe, 0xbe, 0xce, 0x51, 0xbd, 
	0x71, 0x14, 0xd0, 0x3d, 0x5f, 0x79, 0x4b, 0xbe, 0x5e, 0x2d, 0x59, 0x3e, 
	0x25, 0x3c, 0x82, 0xbe, 0xff, 0xed, 0x57, 0xbe, 0x00, 0x64, 0x2a, 0x3a, 
	0x39, 0x08, 0xd8, 0xbc, 0x81, 0x78, 0xa0, 0x3d, 0xc5, 0x45, 0x6b, 0xbe, 
	0xe9, 0xea, 0xff, 0x3d, 0x73, 0xf7, 0xda, 0xbc, 0x9d, 0xfd, 0x47, 0xbe, 
	0xc6, 0xff, 0xca, 0xbe, 0x80, 0x1e, 0xda, 0xbd, 0x2a, 0x23, 0x75, 0xbd, 
	0xf1, 0x67, 0x86, 0x3e, 0xca, 0xa6, 0x09, 0x3f, 0x0c, 0x33, 0x78, 0xbc, 
	0x31, 0x1e, 0x10, 0x3f, 0x95, 0x02, 0x82, 0x3e, 0xb8, 0xb4, 0x01, 0x3f, 
	0x3e, 0xc8, 0xda, 0x3e, 0x60, 0x1d, 0x87, 0x3e, 0x2f, 0xd0, 0x8f, 0x3e, 
	0xfc, 0xb1, 0x3d, 0x3e, 0x98, 0x7e, 0x82, 0x3d, 0xc8, 0x80, 0xfa, 0xbc, 
	0x8f, 0xb0, 0x99, 0xbe, 0x3c, 0x79, 0x8b, 0x3e, 0xa8, 0xe4, 0x78, 0x3e, 
	0xfe, 0x16, 0x32, 0x3f, 0x11, 0xbf, 0x2c, 0x3e, 0x60, 0x96, 0xe4, 0x3e, 
	0xc5, 0xfc, 0xd7, 0x3e, 0x4b, 0x86, 0xfe, 0x3e, 0x41, 0xeb, 0x9b, 0x3e, 
	0x44, 0x1e, 0x35, 0x3f, 0xae, 0x18, 0x8f, 0x3e, 0x74, 0x98, 0xdd, 0x3e, 
	0x02, 0xcd, 0x89, 0x3e, 0x1d, 0x4a, 0xa8, 0x3e, 0x30, 0xa8, 0xd8, 0x3e, 
	0x44, 0xa3, 0x1f, 0x3f, 0x6a, 0x7d, 0x59, 0xbb, 0x50, 0x01, 0xac, 0x3e, 
	0x27, 0x7b, 0xe7, 0x3e, 0x56, 0x3e, 0x04, 0x3f, 0x36, 0xa7, 0x51, 0x3f, 
	0x5c, 0xca, 0xdf, 0x3e, 0xa7, 0x7e, 0x88, 0x3e, 0x91, 0xb9, 0xbc, 0x3e, 
	0x0c, 0xa3, 0x32, 0x3e, 0x0f, 0x1f, 0xf4, 0x3e, 0x4f, 0x73, 0xf3, 0x3e, 
	0xe4, 0x2a, 0xc5, 0x3e, 0x8a, 0xd0, 0xa4, 0x3d, 0x26, 0x62, 0x0b, 0x3f, 
	0x93, 0x5d, 0x93, 0xbc, 0x9d, 0x67, 0x66, 0x3e, 0x3d, 0xa0, 0x16, 0x3e, 
	0xa5, 0xba, 0x8b, 0x3e, 0xee, 0xa8, 0xe6, 0x3e, 0xd2, 0xd6, 0xf0, 0x3d, 
	0xff, 0x46, 0xc4, 0xbd, 0x5a, 0x61, 0xd0, 0x3e, 0x58, 0xfe, 0xdc, 0x3e, 
	0x03, 0x18, 0xdc, 0x3e, 0xc6, 0xbf, 0x48, 0x3e, 0x4d, 0x48, 0x7e, 0x3f, 
	0x54, 0x9b, 0x3c, 0x3e, 0xc1, 0xab, 0xe0, 0x3e, 0x70, 0xc3, 0xc5, 0x3d, 
	0xa9, 0x61, 0x27, 0x3e, 0x54, 0x21, 0x3d, 0x3e, 0xfe, 0xf1, 0x99, 0x3d, 
	0x94, 0xe6, 0xda, 0x3e, 0xd2, 0x72, 0x8d, 0x3d, 0x54, 0xf2, 0xf1, 0x3e, 
	0xea, 0xb6, 0x08, 0x3f, 0xfa, 0x6c, 0x04, 0x3f, 0x1e, 0xcb, 0x56, 0x3f, 
	0xdd, 0x38, 0xe2, 0xbd, 0xcc, 0x96, 0xb5, 0x3e, 0xef, 0x0e, 0x9e, 0x3e, 
	0x5a, 0x77, 0x21, 0xbe, 0xc0, 0x9f, 0xe6, 0x3e, 0xdb, 0x95, 0x23, 0x3e, 
	0x52, 0x0b, 0x6c, 0x3e, 0x22, 0x77, 0x5a, 0x3e, 0xad, 0xad, 0xf0, 0x3e, 
	0x39, 0x98, 0x37, 0x3d, 0x60, 0x3a, 0x07, 0xbf, 0x37, 0x7b, 0x44, 0x3f, 
	0xb5, 0x9f, 0x21, 0x3e, 0xf6, 0xaa, 0x2d, 0x3f, 0x45, 0x5d, 0xbe, 0x3e, 
	0x7a, 0xfc, 0x16, 0xbc, 0x33, 0xfa, 0xd1, 0x3e, 0xf9, 0xc6, 0xf1, 0xbd, 
	0xed, 0xd1, 0x11, 0x3f, 0x57, 0x1f, 0x40, 0x3f, 0x83, 0xe1, 0x24, 0x3c, 
	0xd9, 0x67, 0x95, 0x3e, 0x24, 0x1c, 0x4e, 0x3f, 0xd1, 0x7f, 0xed, 0x3e, 
	0x32, 0xf6, 0x94, 0x3d, 0x67, 0xa6, 0x97, 0x3f, 0x42, 0x91, 0x22, 0x3d, 
	0xa9, 0x98, 0x58, 0xbd, 0xa1, 0x43, 0x61, 0x3e, 0x7d, 0xea, 0xb7, 0x3e, 
	0xde, 0xbd, 0x39, 0x3f, 0x05, 0x70, 0xa6, 0x3e, 0xa6, 0x7a, 0xe4, 0x3e, 
	0x3e, 0x4d, 0x04, 0x3f, 0x7d, 0x6d, 0xc4, 0x3d, 0xf0, 0x1c, 0x9a, 0xbd, 
	0xcb, 0xde, 0x33, 0x3e, 0xa0, 0xe6, 0x73, 0x3d, 0x55, 0xe4, 0x46, 0x3e, 
	0xbe, 0x1a, 0xe8, 0x3e, 0x35, 0x42, 0x94, 0x3d, 0xf7, 0xe8, 0x9f, 0x3e, 
	0x55, 0x1d, 0x80, 0xbd, 0x1c, 0x07, 0x0b, 0x3f, 0x6f, 0x6b, 0xa2, 0x3e, 
	0x87, 0x65, 0xa0, 0x3e, 0xa1, 0x9a, 0x80, 0x3e, 0x30, 0xcf, 0x94, 0x3e, 
	0x8a, 0x50, 0x2b, 0x3e, 0x28, 0xe3, 0x1a, 0x3d, 0x6c, 0xe3, 0x8d, 0x3e, 
	0x79, 0x15, 0x69, 0x3f, 0xd4, 0x17, 0x24, 0xbe, 0x99, 0x54, 0x6b, 0x3d, 
	0xb5, 0x43, 0xc4, 0x3e, 0x50, 0x54, 0x98, 0x3e, 0xdc, 0x75, 0x1e, 0x3e, 
	0x3d, 0x23, 0x1f, 0x3f, 0xc5, 0x5d, 0x29, 0x3e, 0xfe, 0x8b, 0x0f, 0x3f, 
	0xd6, 0x38, 0xdd, 0x3e, 0xe1, 0xda, 0xe2, 0x3e, 0x05, 0x0a, 0x65, 0x3f, 
	0x51, 0x31, 0xee, 0xbd, 0xbb, 0xb1, 0x7d, 0x3d, 0x0f, 0xad, 0x7f, 0xbe, 
	0x73, 0x6f, 0x61, 0xbe, 0xec, 0x36, 0xa4, 0xbe, 0xc5, 0xfa, 0x90, 0xbe, 
	0x3a, 0x52, 0x2a, 0xbe, 0x5b, 0x0b, 0x8a, 0x3e, 0x3e, 0x32, 0x92, 0xbe, 
	0x96, 0xbf, 0x8e, 0x3e, 0x97, 0xef, 0x62, 0xbe, 0x25, 0x55, 0xc9, 0x3d, 
	0x1c, 0x58, 0xd0, 0x3d, 0x84, 0x80, 0x80, 0xbe, 0xc7, 0x4d, 0x66, 0x3e, 
	0xd7, 0x47, 0x6c, 0x3c, 0xf4, 0xc2, 0xd7, 0xbc, 0xfb, 0xd9, 0xa4, 0xbe, 
	0x5d, 0x77, 0xba, 0x3c, 0xd1, 0x63, 0xa6, 0x3e, 0x8e, 0x05, 0x13, 0x3e, 
	0x95, 0x18, 0x37, 0xbe, 0x2a, 0xf8, 0x44, 0xbe, 0x05, 0x59, 0x80, 0x3e, 
	0x7f, 0x73, 0x88, 0xbe, 0x85, 0x17, 0x0f, 0xbe, 0x26, 0x4d, 0xec, 0x3c, 
	0xea, 0xb1, 0x21, 0xbe, 0x9c, 0xc5, 0xc1, 0x3d, 0xb8, 0x18, 0x40, 0x3d, 
	0xad, 0xff, 0xa5, 0x3e, 0x40, 0x44, 0x37, 0xbe, 0xb0, 0x7f, 0x1b, 0xbe, 
	0x97, 0x9e, 0x0c, 0x3e, 0xbf, 0x84, 0x0b, 0xbe, 0xdd, 0x0d, 0xe3, 0x3d, 
	0xe6, 0x24, 0x9b, 0xbe, 0x4f, 0xdf, 0x2b, 0xbe, 0xa2, 0xb6, 0x87, 0x3d, 
	0x7c, 0x7e, 0xc1, 0xbe, 0x14, 0x1c, 0xf2, 0x3d, 0xbb, 0x65, 0x02, 0x3d, 
	0x48, 0xd2, 0x8d, 0x3e, 0x20, 0x2f, 0x65, 0x3e, 0x27, 0x22, 0x80, 0xbd, 
	0xbe, 0x6a, 0x06, 0xbe, 0x4f, 0x7f, 0x75, 0xbe, 0xbe, 0x38, 0x5b, 0xbe, 
	0xa5, 0x05, 0x0b, 0x3e, 0x72, 0x26, 0x10, 0x3b, 0xcf, 0x2a, 0xa7, 0x3d, 
	0xe0, 0x56, 0x2b, 0x3e, 0x7e, 0x76, 0xa8, 0xbd, 0xc2, 0xfa, 0x58, 0x3e, 
	0x97, 0xe0, 0xe4, 0x3d, 0x33, 0xae, 0x4b, 0xbe, 0xac, 0x8f, 0xc3, 0xbe, 
	0x62, 0x19, 0x85, 0xbe, 0x41, 0x81, 0x6e, 0x3e, 0xfd, 0x79, 0xbf, 0x3d, 
	0x36, 0xbc, 0x36, 0x3a, 0xaf, 0x46, 0x31, 0x3d, 0x67, 0x9d, 0x71, 0xbe, 
	0x5b, 0x9d, 0x0c, 0xbe, 0x6f, 0x44, 0x26, 0x3d, 0xdb, 0x57, 0x94, 0xbe, 
	0x3f, 0xaf, 0x53, 0xbe, 0xcc, 0x50, 0x08, 0xbe, 0x42, 0x1f, 0xf5, 0x3d, 
	0x1d, 0xd4, 0x4b, 0x3e, 0x43, 0x09, 0x9f, 0xbe, 0x8d, 0x5d, 0x2a, 0xbd, 
	0x41, 0x20, 0x88, 0xbe, 0x25, 0xac, 0x93, 0xbc, 0x67, 0xaa, 0x4d, 0xbe, 
	0x55, 0x72, 0xe9, 0xbc, 0x8d, 0xa8, 0x2d, 0xbe, 0x14, 0x22, 0x44, 0x3e, 
	0xa7, 0x09, 0x27, 0x3d, 0xac, 0x03, 0x3c, 0xbe, 0xec, 0x3b, 0x9f, 0xbe, 
	0x68, 0xd5, 0x84, 0x3e, 0x64, 0xdc, 0x8d, 0xbe, 0x3e, 0x3a, 0xaa, 0x3d, 
	0x13, 0x6d, 0x08, 0xbe, 0xe0, 0x4f, 0x73, 0xbd, 0x6b, 0xed, 0x26, 0x3e, 
	0x9e, 0x8d, 0x04, 0x3e, 0xb3, 0x22, 0x81, 0xbe, 0x76, 0x02, 0x7c, 0x3e, 
	0x62, 0xf4, 0x13, 0xbe, 0x66, 0x49, 0x8a, 0xbe, 0x69, 0xf9, 0x8d, 0xbe, 
	0x6e, 0xa7, 0x81, 0xbe, 0xab, 0xad, 0x51, 0x3e, 0xd0, 0xbb, 0x91, 0x3e, 
	0x88, 0x47, 0xcb, 0x3c, 0x24, 0x35, 0x0f, 0xbe, 0x5a, 0x73, 0x9d, 0xbe, 
	0xc4, 0x93, 0xc0, 0xbe, 0x7d, 0x48, 0x90, 0x3e, 0xcf, 0x3d, 0xc8, 0x3c, 
	0xf7, 0x92, 0xa5, 0xbe, 0x0c, 0xf2, 0x43, 0xbe, 0x0f, 0x20, 0x16, 0x3e, 
	0xb9, 0x95, 0xf6, 0xbd, 0xe9, 0xa5, 0xe6, 0x3d, 0x88, 0xd3, 0xc6, 0xbe, 
	0x02, 0x99, 0x1f, 0x3d, 0x88, 0xaf, 0xe1, 0x3d, 0x7d, 0x51, 0xc0, 0x3c, 
	0x1a, 0x12, 0x99, 0x3d, 0xef, 0x94, 0x59, 0x3e, 0x5f, 0x8d, 0xf7, 0x3d, 
	0xc1, 0x99, 0x82, 0xbe, 0x94, 0x2e, 0xc0, 0xbe, 0xab, 0x3e, 0x99, 0x3d, 
	0x05, 0x08, 0x20, 0xbe, 0x45, 0xf7, 0x43, 0x3e, 0xca, 0x93, 0x1f, 0xbe, 
	0xcf, 0xec, 0x42, 0xbe, 0x7b, 0x45, 0x07, 0x3d, 0xb4, 0x85, 0x25, 0xbe, 
	0x55, 0x15, 0x8d, 0xbe, 0xc4, 0x28, 0x3b, 0xbd, 0x23, 0x60, 0x17, 0x3e, 
	0x40, 0x41, 0x2e, 0x3c, 0xda, 0x7b, 0x2a, 0xbe, 0xbe, 0xfe, 0xff, 0xff, 
	0x10, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 
	0x2c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 
	0x20, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x6d, 0x6f, 0x64, 0x65, 
	0x6c, 0x5f, 0x35, 0x2f, 0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x36, 0x2f, 
	0x4d, 0x61, 0x74, 0x4d, 0x75, 0x6c, 0x00, 0x00, 0xbc, 0xfe, 0xff, 0xff, 
	0xa2, 0xff, 0xff, 0xff, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 
	0x46, 0x60, 0x33, 0x3e, 0x0e, 0xff, 0xff, 0xff, 0x10, 0x00, 0x00, 0x00, 
	0x03, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x38, 0x00, 0x00, 0x00, 
	0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00, 
	0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x5f, 0x35, 0x2f, 0x64, 0x65, 0x6e, 0x73, 
	0x65, 0x5f, 0x37, 0x2f, 0x42, 0x69, 0x61, 0x73, 0x41, 0x64, 0x64, 0x2f, 
	0x52, 0x65, 0x61, 0x64, 0x56, 0x61, 0x72, 0x69, 0x61, 0x62, 0x6c, 0x65, 
	0x4f, 0x70, 0x00, 0x00, 0x60, 0xff, 0xff, 0xff, 0x00, 0x00, 0x06, 0x00, 
	0x08, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 
	0x40, 0x00, 0x00, 0x00, 0x16, 0x1a, 0x82, 0x3e, 0xc3, 0x8a, 0xc1, 0xbc, 
	0xe3, 0x82, 0xf9, 0x3c, 0xcd, 0x5c, 0xa9, 0x3e, 0x9f, 0x3a, 0xd2, 0xbc, 
	0x43, 0xac, 0xa6, 0x3e, 0x20, 0x36, 0x68, 0xbd, 0x2f, 0x07, 0x55, 0xbd, 
	0x2a, 0x4e, 0x59, 0x3e, 0x75, 0x7e, 0x4b, 0x3e, 0xb6, 0xee, 0xa0, 0x3e, 
	0xdf, 0x55, 0x85, 0x3e, 0x92, 0xdd, 0x15, 0xbd, 0x4e, 0x98, 0x48, 0xbd, 
	0xc9, 0x8f, 0x5d, 0xbd, 0x6f, 0xd0, 0x3f, 0xbd, 0xae, 0xff, 0xff, 0xff, 
	0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 
	0x3c, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 
	0x26, 0x00, 0x00, 0x00, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x5f, 0x35, 0x2f, 
	0x64, 0x65, 0x6e, 0x73, 0x65, 0x5f, 0x36, 0x2f, 0x42, 0x69, 0x61, 0x73, 
	0x41, 0x64, 0x64, 0x2f, 0x52, 0x65, 0x61, 0x64, 0x56, 0x61, 0x72, 0x69, 
	0x61, 0x62, 0x6c, 0x65, 0x4f, 0x70, 0x00, 0x00, 0x04, 0x00, 0x06, 0x00, 
	0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x14, 0x00, 0x04, 0x00, 
	0x00, 0x00, 0x08, 0x00, 0x0c, 0x00, 0x10, 0x00, 0x0e, 0x00, 0x00, 0x00, 
	0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 
	0x1c, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 
	0x20, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x69, 0x6e, 0x70, 0x75, 
	0x74, 0x5f, 0x34, 0x00, 0xfc, 0xff, 0xff, 0xff, 0x04, 0x00, 0x04, 0x00, 
	0x04, 0x00, 0x00, 0x00
};
const int lat2_32_seq10_len = 3220;