#include <metal_stdlib>
using namespace metal;

// Optimized compute kernel for high-throughput pixel manipulation on M5
kernel void imageThreshold(texture2d<float, access::read>  inTexture  [[texture(0)]],
                           texture2d<float, access::write> outTexture [[texture(1)]],
                           constant float &threshold                  [[buffer(0)]],
                           uint2 gid                                  [[thread_position_in_grid]])
{
    // Safety guard to ensure threads outside image boundaries do not execute
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }

    // Direct read from texture cache memory space
    float4 inColor = inTexture.read(gid);

    // Calculate luminance using ITU-R BT.601 coefficients
    float luminance = dot(inColor.rgb, float3(0.299f, 0.587f, 0.114f));

    // Execute absolute binary threshold evaluation
    float3 outColor = (luminance >= threshold) ? float3(1.0f) : float3(0.0f);

    // Write back to the output texture pipeline preserving original alpha channels
    outTexture.write(float4(outColor, inColor.a), gid);
}
