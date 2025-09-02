<!-- Bilingual Links -->
English | [中文 / Chinese](README-zh.md)

# ComfyUI FlexAI Plugin

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/your-repo/Comfyui-flexai)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A modern, unified ComfyUI plugin for OpenAI-compatible APIs with enhanced debugging and dual-mode image processing capabilities.

## ✨ Key Features

### 🌐 Multiple API Sources Support
- **Flexible Configuration**: Support multiple API providers via `.env` file
- **Dynamic Selection**: Switch between providers without restarting ComfyUI
- **Auto-Detection**: System automatically detects and populates provider dropdown
- **Wide Compatibility**: OpenAI, Anthropic, custom endpoints, and more

### 🖼️ OpenAI Image Node (`flexai:openai_image`)

**Dual-Mode Operation:**
- **Edit Mode**: Provide 1-4 images → Uses `images.edit` API
- **Generate Mode**: No images → Uses `images.generate` API

**Enhanced Features:**
- **Smart Response Handling**: Supports both base64 and URL responses
- **Auto Image Download**: Downloads and converts URL responses automatically
- **Enhanced Debug Mode**: Detailed request/response logging with timing
- **English Error Display**: Clear error messages without font issues
- **Comprehensive Error Handling**: Safety rejection guidance and visual feedback

### 💬 OpenAI Text Node (`flexai:openai_text`)

**Multimodal Text Generation:**
- Pure text or vision-language understanding (VQA)
- Support 1-4 reference images with auto-scaling
- Streaming and non-streaming modes
- **Debug Mode**: Complete JSON logging for all operations

### 🔧 Enhanced Debugging System

**New Debug Features:**
- **Detailed Timing**: Precise timing for each processing stage
- **Network Monitoring**: HTTP request/response tracking
- **Progress Indicators**: Visual feedback during long operations
- **Error Analysis**: Smart error categorization with solutions
- **English Error Images**: All error displays use English to avoid font issues

## Quick Start

### Installation

1. Clone to ComfyUI custom nodes:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-repo/Comfyui-flexai.git
   ```

2. Install dependencies:
   ```bash
   cd Comfyui-flexai
   pip install -r requirements.txt
   ```

3. Configure providers (see Configuration section)
4. Restart ComfyUI

## Configuration

Create a `.env` file in the plugin root directory.

### Single Provider
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # Optional
```

### Multiple Providers (Recommended)
```bash
# Define provider list
OPENAI_PROVIDERS=openai,anthropic,custom

# OpenAI
OPENAI_API_KEY_openai=sk-your-openai-key
OPENAI_API_BASE_openai=https://api.openai.com/v1

# Anthropic (via OpenAI-compatible endpoint)
OPENAI_API_KEY_anthropic=sk-your-anthropic-key  
OPENAI_API_BASE_anthropic=https://api.anthropic.com/v1

# Custom endpoint
OPENAI_API_KEY_custom=your-custom-key
OPENAI_API_BASE_custom=https://your-api.example.com/v1
```

## Node Parameters

### Image Node (`flexai:openai_image`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | Choice | API provider selection |
| `model` | String | Model name (e.g., `dall-e-3`, `dall-e-2`) |
| `prompt` | String | Generation/editing prompt |
| `image_1-4` | Image | Optional images (edit mode if any provided) |
| `size` | String | Output size (e.g., `1024x1024`) |
| `debug` | Boolean | **Enable detailed debug logging** |

### Text Node (`flexai:openai_text`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | Choice | API provider selection |
| `model` | String | Model name (e.g., `gpt-4o`, `gpt-3.5-turbo`) |
| `system_prompt` | String | System message |
| `user_prompt` | String | User message |
| `image_1-4` | Image | Optional reference images |
| `max_tokens` | Integer | Maximum response tokens |
| `temperature` | Float | Sampling temperature (0.0-1.0) |
| `stream` | Boolean | Enable streaming mode |
| `debug` | Boolean | **Enable detailed debug logging** |

## Debug Mode Features

Enable debug mode (`debug=True`) for comprehensive logging:

### 🔍 API Request/Response Tracking
```
============================================================
[DEBUG] 🚀 开始图片生成请求
[DEBUG] ⏰ 请求时间: 2024-09-01 14:30:25
[DEBUG] 📝 提交到OpenAI Images API的原生JSON数据:
{
  "model": "dall-e-3",
  "prompt": "A cute cat",
  "size": "1024x1024",
  "response_format": "b64_json"
}
============================================================
[DEBUG] 📡 正在发送API请求...
[DEBUG] 💡 生成时间通常在10-60秒之间，请耐心等待...
```

### ⏱️ Detailed Timing Analysis
```
[DEBUG] 🎉 图片生成流程完成!
[DEBUG] ⏱️  总耗时: 23.45 秒
[DEBUG]    ├─ API调用: 22.10 秒
[DEBUG]    └─ 数据解码: 1.35 秒
```

### 🌐 Network Download Monitoring
```
[DEBUG] 🌐 开始下载图片
[DEBUG] 📡 发送HTTP GET请求...
[DEBUG] ✅ 下载成功!
[DEBUG] ⏱️  下载耗时: 3.24 秒
[DEBUG] 📏 下载数据大小: 1,234,567 字节 (1.2 MB)
[DEBUG] 🖼️  检测到PNG格式图片
```

### 🚨 Smart Error Analysis
- **API Configuration Issues**: Automatic detection and solutions
- **Network Problems**: Detailed connection diagnostics  
- **Model Compatibility**: Supported model recommendations
- **Safety Rejections**: Content policy guidance

## Troubleshooting Guide

### Common Issues & Solutions

| Error Type | Symptoms | Solution |
|------------|----------|----------|
| **API Key Problem** | "API key not configured" | Check `.env` file configuration |
| **Network Issues** | "Unable to connect" | Check internet connection/proxy |
| **Unsupported Model** | "not supported model" | Use `dall-e-3` or `dall-e-2` |
| **Safety Rejection** | "safety system rejected" | Modify prompt content |
| **Timeout** | Long wait times | Increase timeout or check API status |

### Debug Mode Benefits
1. **Performance Analysis**: Identify bottlenecks in processing
2. **Network Diagnostics**: Track download speeds and failures
3. **Error Diagnosis**: Get specific error types and solutions
4. **API Monitoring**: See exact requests and responses
5. **Progress Tracking**: Understand processing stages

## Technical Details

### Enhanced Error Handling
- **English Error Images**: All error messages display in English to avoid font issues
- **Smart Error Translation**: Automatic translation of common error messages
- **Multi-System Font Support**: Compatible across macOS/Linux/Windows
- **Detailed Error Context**: Timestamps and formatted error information

### Response Format Compatibility
- **Base64 Response**: Direct processing of various base64 field formats
- **URL Response**: Automatic download and conversion
- **Smart Fallback**: Seamless handling across different API providers
- **Format Detection**: Automatic PNG/JPEG format identification

### Architecture
- **Modern SDK**: Built on OpenAI Python SDK 1.x
- **Multi-Provider Support**: Flexible API source configuration
- **Clean Structure**: Unified `flexai:` namespace
- **Modular Design**: Separate image and text processing utilities

### File Structure
```
Comfyui-flexai/
├── __init__.py                 # Plugin registration
├── provider_config.py          # Multi-provider management
├── nodes/
│   ├── flexai/
│   │   ├── openai_image.py    # Enhanced image node with debug
│   │   └── openai_text.py     # Enhanced text node with debug
│   └── utils/
│       ├── openai_standard.py # API utilities with logging
│       └── images.py          # Image processing utilities
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Best Practices

### For Image Generation
- Use `dall-e-3` for highest quality (slower)
- Use `dall-e-2` for faster generation
- Enable debug mode when troubleshooting
- Keep prompts under safety policy guidelines

### For Performance
- Monitor debug logs for timing bottlenecks
- Use appropriate image sizes (1024x1024 recommended)
- Consider network speed for URL-based responses
- Set reasonable timeout values

### For Debugging
- Always enable `debug=True` when experiencing issues
- Check console output for detailed diagnostics
- Use timing information to identify slow components
- Share debug logs when reporting issues

## Support

- **Issues**: Report bugs via GitHub Issues with debug logs
- **Feature Requests**: Submit via GitHub Discussions  
- **Documentation**: Check README-zh.md for Chinese version
- **Debug Help**: Enable debug mode and share console output

## Update History

### v1.0.0 (2024-09-02)
- Enhanced API response debugging and safety checks for None data
- Improved debugging features with Chinese font issue fixes
- Added advanced timeout controls and progress simulation
- Implemented URL response support with comprehensive debug logging
- Multi-provider configuration support
- Initial stable release with dual-mode image processing

---

Built with ❤️ for the ComfyUI community
