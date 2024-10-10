
int main() {

    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());

    context.create();
    pipeline.setLaunchVariableName("params");
    Module module = pipeline.createModuleFromOptixIr(context, "kernels.cu.optixir");

    ProgramGroup raygen = pipeline.createRaygenProgram(context, module, "__raygen__rg");
    ProgramGroup miss = pipeline.createMissProgram(ctx, module, "__miss__envmap");
    ProgramGroup mesh_prg = pipeline.createHitgroupProgram(ctx, module, "__closesthit__mesh");

}