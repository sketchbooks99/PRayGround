#include "util.cuh"

static __forceinline__ __device__ Vec3f reinhardToneMap(const Vec3f& color, const float white)
{
    const float l = luminance(color);
    return (color * 1.0f) / (1.0f + l / white);
}

// Simple path tracer w/o MIS, NEE
extern "C" __device__ void __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const Vec3ui idx(optixGetLaunchIndex());
    uint32_t seed = tea<4>(idx.y() * params.width + idx.x(), frame);

    Vec3f result(0.0f);
    Vec3f normal(0.0f);
    float p_depth = 0.0f;
    Vec3f albedo(0.0f);

    int i = params.samples_per_launch;

    do
    {
        const Vec2f jitter = UniformSampler::get2D(seed) - 0.5f;
        const Vec2f res(params.width, params.height);
        const Vec2f coord(idx.x(), idx.y());
        const Vec2f d = 2.0f * ((coord + jitter) / res) - 1.0f;

        Vec3f ro, rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        Vec3f throughput(1.0f);
        Vec3f radiance(0.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = Vec3f(0.0f);
        si.albedo = Vec3f(0.0f);
        si.trace_terminate = false;
        si.radiance_evaled = false;

        float tmax = raygen->camera.farclip / dot(rd, normalize(raygen->camera.lookat - ro));

        int depth = 0;
        for ( ;; ) {

            if (depth >= params.max_depth)
                break;

            trace(params.handle, ro, rd, 0.01f, tmax, 0, &si);

            //if (si.trace_terminate) {
            //    result += si.emission;
            //    break;
            //}

            // Get emission from area emitter
            if ( si.surface_info.type == SurfaceType::AreaEmitter )
            {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data);
                //result += si.emission * throughput;
                //result += si.emission;

                if (depth == 0) {
                    albedo = si.albedo;
                    Vec3f op = si.p - ro;
                    float op_length = length(si.p - ro);
                    p_depth = dot(normalize(op), normalize(raygen->camera.lookat - ro)) * op_length;
                    p_depth = p_depth / raygen->camera.farclip;
                    normal = si.shading.n;
                }

                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info.type & SurfaceType::Delta))
            {
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.sample, &si, si.surface_info.data);
                
                // Evaluate bsdf
                Vec3f bsdf_val = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data);
                throughput *= bsdf_val;
            }
            // Rough surface sampling with applying MIS
            else if ( +(si.surface_info.type & (SurfaceType::Rough | SurfaceType::Diffuse)) )
            {
                int light_id = rndInt(si.seed, 0, params.num_lights - 1);
                AreaEmitterInfo light = params.lights[light_id];

                // Explicit light sampling
                LightInteraction li = optixDirectCall<LightInteraction, const AreaEmitterInfo&, SurfaceInteraction*>
                    (light.sample_id, light, &si);

                const float dist_to_light = length(li.p - si.p);
                const Vec3f to_light = normalize(li.p - si.p);
                const float NdotL = dot(si.shading.n, to_light);    // Cosine between surface normal and light vector

                float LNdotL = -dot(li.n, to_light);                // Cosine between light normal and light vector
                if (light.twosided)
                {
                    li.n = faceforward(li.n, to_light, li.n);
                    LNdotL = dot(li.n, to_light);
                }

                if (light_id == 1)
                {
                    printf("Contribution from sphere: NdotL: %f, LNdotL: %f, pdf: %f\n", NdotL, LNdotL, li.pdf);
                }

                bool is_contributed = false;
                if (NdotL > 0.0f && LNdotL > 0.0f)
                {
                    const bool occluded = traceShadow(params.handle, si.p, to_light, 0.001f, dist_to_light - 0.001f);
                    if (!occluded)
                    {
                        SurfaceInteraction light_si;
                        light_si.p = li.p;
                        light_si.shading.n = li.n;
                        light_si.shading.uv = li.uv;
                        light_si.wo = to_light;
                        
                        // Get emittance from sampled area emitter
                        optixDirectCall<void, SurfaceInteraction*, void*>(
                            light.surface_info.callable_id.bsdf, &light_si, light.surface_info.data);

                        Vec3f contrib_from_light = light_si.emission * NdotL * LNdotL / (math::pi * li.pdf);

                        si.wi = to_light;

                        float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                            si.surface_info.callable_id.pdf, &si, si.surface_info.data);

                        // Evaluate BSDF
                        Vec3f bsdf_val = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                            si.surface_info.callable_id.bsdf, &si, si.surface_info.data);

                        radiance += contrib_from_light;
                        throughput *= bsdf_val / bsdf_pdf;

                        is_contributed = true;
                    }
                }

                // Importance sampling according to the BSDF
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.sample, &si, si.surface_info.data);

                if (!is_contributed)
                {
                    // Evaluate PDF depends on BSDF
                    float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                        si.surface_info.callable_id.pdf, &si, si.surface_info.data);

                    // Evaluate BSDF
                    Vec3f bsdf_val = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                        si.surface_info.callable_id.bsdf, &si, si.surface_info.data);

                    throughput *= bsdf_val / bsdf_pdf;
                }

            }

            result += si.emission;
            result += radiance * throughput;

            if (si.trace_terminate || depth >= params.max_depth)
                break;

            if (depth == 0) {
                albedo += si.albedo;
                Vec3f op = si.p - ro;
                float op_length = length(si.p - ro);
                p_depth += (dot(normalize(op), normalize(raygen->camera.lookat - ro)) * op_length) / raygen->camera.farclip;
                normal += si.shading.n;
            }

            // Make tmax large except for when the primary ray
            tmax = 1e16f;
            
            ro = si.p;
            rd = si.wi;

            ++depth;
        }
    } while (--i);

    const uint32_t image_index = idx.y() * params.width + idx.x();

    //if (!result.isValid()) result = Vec3f(0.0f);

    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);

    if (frame > 0)
    {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_color_prev(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = Vec4f(accum_color, 1.0f);
    Vec3u color = make_color(reinhardToneMap(accum_color, params.white));
    params.result_buffer[image_index] = Vec4u(color, 255);
    params.normal_buffer[image_index] = normal;
    params.albedo_buffer[image_index] = albedo;
    params.depth_buffer[image_index] = p_depth == 0.0f ? 1.0f : p_depth;
}