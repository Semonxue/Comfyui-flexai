<!-- Bilingual Links -->
English | [中文 / Chinese](README-zh.md)

# ComfyUI FlexAI Plugin

A modern, unified ComfyUI plugin for OpenAI-compatible APIs with dual-mode image processing capabilities.

## Features

### 🌐 Multiple API Sources Support
- **Flexible Configuration**: Support configuring multiple API providers in `.env` file
- **Dynamic Selection**: Nodes can choose different API sources for calls
- **Auto-Detection**: System automatically detects configured providers and populates dropdown menus
- **Seamless Switching**: Switch between different providers without restarting

### 🖼️ OpenAI Image Node (`flexai:openai_image`)

**Dual-Mode Operation:**
- **Edit Mode**: Provide 1-4 images → Uses `images.edit` API
- **Generate Mode**: Provide no images → Uses `images.generate` API

**Key Features:**
- Supports 1-4 simultaneous image inputs for editing
- Automatic mode detection based on image inputs
- Modern OpenAI Python SDK (>=1.0) integration
- **Dual Response Format Support**: Automatically handles both base64 and URL responses
- **Auto Image Download**: When API returns URLs, automatically downloads and converts images
- Comprehensive error handling with visual feedback
- Safety system rejection guidance
- **Enhanced Debug Mode**: Detailed JSON request/response logging

### 💬 OpenAI Text Node (`flexai:openai_text`)

**Multimodal Text Generation:**
- Pure text or vision-language understanding (VQA)
- Support for 1-4 reference images with automatic downscaling
- Streaming and non-streaming modes
- Auto-fallback for unsupported models

**Advanced Features:**
- OpenAI SDK compatibility (1.x preferred, 0.x fallback)
- Usage statistics tracking
- Reproducible generation with seed control
- Smart image preprocessing (≤1024px longest side)
- **Enhanced Debug Mode**: Complete JSON request/response logging for both streaming and non-streaming modes

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

### Basic Usage

**Image Generation:**
```
Add flexai:openai_image node
→ Set provider and model
→ Enter prompt
→ Generate!
```

**Image Editing:**
```
Add flexai:openai_image node
→ Connect image(s) to image_1/2/3/4 inputs
→ Set provider and model  
→ Enter editing prompt
→ Edit!
```

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

### Nano-Banana (Gemini-2.5-Flash-Image-Preview) Support
```bash
# Nano-Banana Configuration Example
OPENAI_API_KEY_nanobanana=your-nanobanana-api-key
OPENAI_API_BASE_nanobanana=https://api.nanobanana.example.com/v1
```

**Supported Models:**
- `gemini-2.5-flash-image-preview`: Call Gemini models through OpenAI-compatible interface, supporting image processing and text generation
- Automatic adaptation to OpenAI SDK calling methods, no additional configuration required

### Auto-Detection
Alternatively, just define keys with suffixes:
```bash
OPENAI_API_KEY_provider1=key1
OPENAI_API_KEY_provider2=key2
```
The system will auto-detect and populate the provider dropdown.

## Node Parameters

### Image Node (`flexai:openai_image`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | Choice | API provider selection |
| `model` | String | Model name (e.g., `dall-e-3`) |
| `prompt` | String | Generation/editing prompt |
| `image_1-4` | Image | Optional images (edit mode if any provided) |
| `size` | String | Output size (e.g., `1024x1024`) |
| `debug` | Boolean | Enable debug logging |

### Text Node (`flexai:openai_text`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | Choice | API provider selection |
| `model` | String | Model name (e.g., `gpt-4o`) |
| `system_prompt` | String | System message |
| `user_prompt` | String | User message |
| `image_1-4` | Image | Optional reference images |
| `max_tokens` | Integer | Maximum response tokens |
| `temperature` | Float | Sampling temperature (0.0-1.0) |
| `stream` | Boolean | Enable streaming mode |
| `debug` | Boolean | Enable debug logging |

## Advanced Usage

### Response Format Compatibility

**Automatic Format Detection:**
- **Base64 Response**: Direct processing of b64_json, b64, base64, or data fields
- **URL Response**: Automatic download and conversion when API returns image URLs
- **Smart Fallback**: Seamless handling across different API providers

**Enhanced Error Handling:**
- Detailed validation of response data types
- Network timeout and retry mechanisms
- Comprehensive diagnostic information in debug mode

### Debug Mode

Enable enhanced debugging (`debug=True`) for detailed insights:
- **API Requests**: Complete JSON parameters sent to API
- **API Responses**: Full JSON responses from API (streaming and non-streaming)
- **Image Processing**: URL download progress and base64 conversion details
- **Error Diagnostics**: Detailed error messages with suggested solutions

### Image Processing Modes

**Pure Generation:**
- Don't connect any images
- Uses `images.generate` endpoint
- Perfect for text-to-image generation

**Single Image Edit:**
- Connect one image to `image_1`
- Uses `images.edit` endpoint
- Great for style transfer, modifications

**Multi-Image Edit:**
- Connect 2-4 images to `image_1`, `image_2`, etc.
- All images sent to `images.edit` as array
- Useful for complex scene editing

### Multi-Provider Usage

**Configuring Multiple Sources:**
- Define multiple providers in the `.env` file
- Each node can independently select API source
- Support for OpenAI, Anthropic, custom endpoints, etc.

**Switching Providers:**
- No need to restart ComfyUI
- Real-time switching between different models and services
- Maintain workflow compatibility

### Error Handling

The plugin includes robust error handling:
- Safety system rejections show helpful guidance
- Network errors display in debug mode
- Failed operations generate error visualization images
- Non-blocking: workflow continues even with failures

### Streaming Text

Enable streaming for real-time text generation:
- Set `stream=True` in text node
- Receive incremental token updates
- Includes usage statistics when supported
- Better UX for long-form generation

## Technical Details

### Architecture
- **Modern SDK**: Built on OpenAI Python SDK 1.x
- **Multi-Provider Support**: Flexible configuration of multiple API sources, supporting OpenAI-compatible endpoints
- **Nano-Banana Integration**: Native support for Gemini-2.5-Flash-Image-Preview and other models
- **Clean Structure**: Unified namespace with `flexai:` prefix
- **Modular Design**: Separate image and text processing

### File Structure
```
Comfyui-flexai/
├── __init__.py                 # Plugin registration
├── provider_config.py          # Multi-provider management
├── nodes/
│   ├── flexai/
│   │   ├── openai_image.py    # Image generation/editing node
│   │   └── openai_text.py     # Text generation node
│   └── utils/
│       ├── openai_standard.py # OpenAI API utilities
│       └── images.py          # Image processing utilities
├── test/
│   └── test_plugin.py         # Plugin tests
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Compatibility

- **ComfyUI**: Any recent version with custom nodes support
- **Python**: 3.8+ (tested with 3.10+)
- **OpenAI SDK**: 1.0+ (preferred) with 0.x fallback
- **APIs**: Any OpenAI-compatible endpoint, including Nano-Banana (Gemini-2.5-Flash-Image-Preview)
- **Providers**: OpenAI, Anthropic, custom endpoints, etc.

## Troubleshooting

### Common Issues

**Plugin not loading:**
- Check `requirements.txt` installation
- Verify `.env` file configuration
- Restart ComfyUI completely

**API errors:**
- Enable `debug=True` for detailed logging
- Verify API keys and endpoints
- Check provider-specific documentation

**Image processing issues:**
- Ensure images are valid ComfyUI tensors
- Check image format compatibility
- Verify model supports image processing

**Safety rejections:**
- Review OpenAI usage policies
- Modify prompt content
- Try different model variants

### Debug Mode

Enable debug mode (`debug=True`) for verbose logging:
- API request/response details
- Image processing steps
- Error stack traces
- Performance metrics

## Testing

Run the test suite:
```bash
python -m test.test_plugin
```

This verifies:
- Plugin loading
- Provider configuration
- Node registration
- Basic functionality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Feature requests and general questions
- **Documentation**: Check README-zh.md for Chinese version

---

Built with ❤️ for the ComfyUI community
