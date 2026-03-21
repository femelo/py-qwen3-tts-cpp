/**
 ********************************************************************************
 * @file    main.cpp
 * @author  [femelo](https://github.com/femelo)
 * @date    2026
 * @brief   Python bindings for [qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) using Pybind11
 *
 * @par
 * COPYRIGHT NOTICE: (c) 2026.  All rights reserved.
 ********************************************************************************
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "qwen3_tts.h"


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

#define DEF_RELEASE_GIL(name, fn, doc) \
    m.def(name, fn, doc, py::call_guard<py::gil_scoped_release>())


namespace py = pybind11;
using namespace qwen3_tts;

PYBIND11_MODULE(_py_qwen3_tts_cpp, m) {
    m.doc() = "Python bindings for Qwen3 TTS engine";

    // 1. Bind the tts_params struct
    py::class_<tts_params>(m, "TtsParams")
        .def(py::init<>())
        .def_readwrite("max_audio_tokens", &tts_params::max_audio_tokens)
        .def_readwrite("temperature", &tts_params::temperature)
        .def_readwrite("top_p", &tts_params::top_p)
        .def_readwrite("top_k", &tts_params::top_k)
        .def_readwrite("n_threads", &tts_params::n_threads)
        .def_readwrite("print_progress", &tts_params::print_progress)
        .def_readwrite("print_timing", &tts_params::print_timing)
        .def_readwrite("repetition_penalty", &tts_params::repetition_penalty)
        .def_readwrite("language_id", &tts_params::language_id);

    // 2. Bind the tts_result struct
    py::class_<tts_result>(m, "TtsResult")
        .def_property_readonly("audio", [](const tts_result &self) {
            return py::array_t<float>(
                static_cast<py::ssize_t>(self.audio.size()),
                self.audio.data()
            );
        })
        .def_readonly("sample_rate", &tts_result::sample_rate)
        .def_readonly("success", &tts_result::success)
        .def_readonly("error_msg", &tts_result::error_msg)
        .def_readonly("t_load_ms", &tts_result::t_load_ms)
        .def_readonly("t_tokenize_ms", &tts_result::t_tokenize_ms)
        .def_readonly("t_encode_ms", &tts_result::t_encode_ms)
        .def_readonly("t_generate_ms", &tts_result::t_generate_ms)
        .def_readonly("t_decode_ms", &tts_result::t_decode_ms)
        .def_readonly("t_total_ms", &tts_result::t_total_ms)
        .def_readonly("mem_rss_start_bytes", &tts_result::mem_rss_start_bytes)
        .def_readonly("mem_rss_end_bytes", &tts_result::mem_rss_end_bytes)
        .def_readonly("mem_rss_peak_bytes", &tts_result::mem_rss_peak_bytes)
        .def_readonly("mem_phys_start_bytes", &tts_result::mem_phys_start_bytes)
        .def_readonly("mem_phys_end_bytes", &tts_result::mem_phys_end_bytes)
        .def_readonly("mem_phys_peak_bytes", &tts_result::mem_phys_peak_bytes);

    // 3. Bind the main Qwen3TTS class
    py::class_<Qwen3TTS>(m, "Qwen3TTS")
        .def(py::init<>())

        // Load from a directory containing both model files
        .def("load_models_from_dir",
             static_cast<bool (Qwen3TTS::*)(const std::string &)>(&Qwen3TTS::load_models),
             py::arg("model_dir"))

        // Load by specifying paths directly
        .def("load_models",
             static_cast<bool (Qwen3TTS::*)(const std::string &, const std::string &)>(&Qwen3TTS::load_models),
             py::arg("tts_model_path"), py::arg("tokenizer_model_path"))

        // Synthesize speech from text
        .def("synthesize",
             &Qwen3TTS::synthesize,
             py::arg("text"), py::arg("params") = tts_params())

        // Synthesize with voice cloning from a WAV file
        .def("synthesize_with_voice_file",
             static_cast<tts_result (Qwen3TTS::*)(const std::string &, const std::string &, const tts_params &)>(&Qwen3TTS::synthesize_with_voice),
             py::arg("text"), py::arg("reference_audio"), py::arg("params") = tts_params())

        // Synthesize with voice cloning from a NumPy array
        .def("synthesize_with_voice_samples",
             [](Qwen3TTS &self, const std::string &text, py::array_t<float> ref_samples, const tts_params &params) {
                 py::buffer_info buf = ref_samples.request();
                 if (buf.ndim != 1) {
                     throw std::runtime_error("Reference audio samples must be a 1D array");
                 }
                 return self.synthesize_with_voice(
                     text,
                     static_cast<float *>(buf.ptr),
                     static_cast<int32_t>(buf.size),
                     params
                 );
             },
             py::arg("text"), py::arg("ref_samples"), py::arg("params") = tts_params())

        // Extract speaker embedding from a NumPy array (for caching / reuse)
        .def("extract_speaker_embedding",
             [](Qwen3TTS &self, py::array_t<float> ref_samples, const tts_params &params) {
                 py::buffer_info buf = ref_samples.request();
                 if (buf.ndim != 1) {
                     throw std::runtime_error("Reference audio samples must be a 1D array");
                 }
                 std::vector<float> embedding;
                 bool ok = self.extract_speaker_embedding(
                     static_cast<float *>(buf.ptr),
                     static_cast<int32_t>(buf.size),
                     embedding,
                     params
                 );
                 if (!ok) {
                     throw std::runtime_error("Failed to extract speaker embedding: " + self.get_error());
                 }
                 return py::array_t<float>(
                     static_cast<py::ssize_t>(embedding.size()),
                     embedding.data()
                 );
             },
             py::arg("ref_samples"), py::arg("params") = tts_params())

        // Synthesize with a pre-computed speaker embedding (skips encoder)
        .def("synthesize_with_embedding",
             [](Qwen3TTS &self, const std::string &text, py::array_t<float> embedding, const tts_params &params) {
                 py::buffer_info buf = embedding.request();
                 if (buf.ndim != 1) {
                     throw std::runtime_error("Embedding must be a 1D array");
                 }
                 return self.synthesize_with_embedding(
                     text,
                     static_cast<float *>(buf.ptr),
                     static_cast<int32_t>(buf.size),
                     params
                 );
             },
             py::arg("text"), py::arg("embedding"), py::arg("params") = tts_params())

        .def("set_progress_callback", &Qwen3TTS::set_progress_callback)
        .def("get_error", &Qwen3TTS::get_error)
        .def("is_loaded", &Qwen3TTS::is_loaded);

    // Utility: load a WAV file and return (audio_array, sample_rate)
    m.def("load_audio_file", [](const std::string &path) {
        std::vector<float> samples;
        int sample_rate = 0;

        bool success = load_audio_file(path, samples, sample_rate);

        if (!success) {
            throw std::runtime_error("Failed to load audio file: " + path);
        }

        py::array_t<float> audio_array(static_cast<py::ssize_t>(samples.size()), samples.data());
        return py::make_tuple(audio_array, sample_rate);
    }, "Loads a WAV file and returns (audio_array, sample_rate)");

    // Utility: save a NumPy array to a WAV file
    m.def("save_audio_file", [](const std::string &path, py::array_t<float> samples, int sample_rate) {
        py::buffer_info buf = samples.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Audio samples must be a 1D array");
        }
        std::vector<float> vec(static_cast<float *>(buf.ptr),
                               static_cast<float *>(buf.ptr) + buf.size);
        bool success = save_audio_file(path, vec, sample_rate);
        if (!success) {
            throw std::runtime_error("Failed to save audio file: " + path);
        }
    }, "Saves a float32 NumPy array as a WAV file");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}
