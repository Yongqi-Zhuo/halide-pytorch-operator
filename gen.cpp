#include "Halide.h"
#include <algorithm>
#include <cassert>
#include <stdio.h>
#include <random>
#include <utility>
#include <vector>
using namespace Halide;

Target find_gpu_target() {
    Target target = get_host_target();

    std::vector<Target::Feature> features_to_try;
    // Uncomment the following lines to also try CUDA:
    features_to_try.push_back(Target::CUDA);
    features_to_try.push_back(Target::UserContext);

    for (Target::Feature f : features_to_try) {
        target = target.with_feature(f);
    }

    assert(host_supports_target_device(target));
    return target;
}

Var x("x"), y("y");

template<typename Input1, typename Input2>
Func get_brighter(const Input1& input1, const Input2& input2) {
    Func brighter("brighter");
    brighter(x, y) = input1(y, x) * input1(y, x) + input2;
    return brighter;
}

int main(int argc, char **argv) {

    load_plugin("autoschedule_li2018");
    Target target = find_gpu_target();
    assert(target.has_gpu_feature());
    auto ext = Internal::get_output_info(target);

    Param<float> offset("offset");
    ImageParam input(type_of<float>(), 2, "input");

    Func brighter = get_brighter(input, cast<float>(offset));

    brighter
        .set_estimate(x, 0, 8)
        .set_estimate(y, 0, 8);
    offset.set_estimate(8.0f);
    input.set_estimates({{0, 8}, {0, 8}});
    
    Pipeline pipeline_brighter(brighter);
    pipeline_brighter.auto_schedule("Li2018", target);

    Module brighter_module = pipeline_brighter.compile_to_module({input, offset}, "brighter", target);


    ImageParam input_other(type_of<float>(), 2, "input_other");
    ImageParam output_grad(type_of<float>(), 2, "output_grad");
    Func input_grad("input_grad");

    Derivative brighter_grad = propagate_adjoints(get_brighter(input_other, offset), output_grad, {
        {0, output_grad.dim(0).extent()},
        {0, output_grad.dim(1).extent()}
    });

    input_grad(x, y) = brighter_grad(input_other)(x, y);

    input_grad
        .set_estimate(x, 0, 8)
        .set_estimate(y, 0, 8);
    output_grad.set_estimates({{0, 8}, {0, 8}});

    Pipeline pipeline_brighter_grad(input_grad);
    pipeline_brighter_grad.auto_schedule("Li2018", target);

    Module brighter_grad_module = pipeline_brighter_grad.compile_to_module({input_other, offset, output_grad}, "brighter_grad", target);

    const auto flagsForModule = [&ext](const std::string& filename) -> std::map<OutputFileType, std::string> { return {
        {OutputFileType::stmt, filename + ext.at(OutputFileType::stmt).extension},
        {OutputFileType::pytorch_wrapper, filename + ext.at(OutputFileType::pytorch_wrapper).extension},
        {OutputFileType::static_library, filename + ext.at(OutputFileType::static_library).extension}
    };};
    brighter_module.compile(flagsForModule("brighter"));
    brighter_grad_module.compile(flagsForModule("brighter_grad"));

    printf("Halide pipeline compiled, but not yet run.\n");

    return 0;
}
