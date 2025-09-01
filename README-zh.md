<!-- 双语链接 -->
[English](README.md) | 中文 / Chinese

# ComfyUI FlexAI 插件

现代化的 ComfyUI 插件，支持 OpenAI 兼容的 API，具备双模式图像处理能力。

## 功能特性

### 🖼️ OpenAI 图片节点 (`flexai:openai_image`)

**双模式运行：**
- **编辑模式**：提供 1-4 张图片 → 使用 `images.edit` API
- **生成模式**：不提供图片 → 使用 `images.generate` API

**核心特性：**
- 支持 1-4 张图片同时输入进行编辑
- 基于图片输入自动检测模式
- 现代 OpenAI Python SDK (>=1.0) 集成
- 全面的错误处理与视觉反馈
- 安全系统拒绝时的友好提示

### 💬 OpenAI 文本节点 (`flexai:openai_text`)

**多模态文本生成：**
- 纯文本或视觉语言理解 (VQA)
- 支持 1-4 张参考图片，自动压缩优化
- 流式和非流式模式
- 不支持模型的自动降级

**高级特性：**
- OpenAI SDK 兼容性 (1.x 优先，0.x 回退)
- 使用统计跟踪
- 使用 seed 的可重现生成
- 智能图片预处理 (≤1024px 最长边)

## 快速开始

### 安装

1. 克隆到 ComfyUI 自定义节点目录：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-repo/Comfyui-flexai.git
   ```

2. 安装依赖：
   ```bash
   cd Comfyui-flexai
   pip install -r requirements.txt
   ```

3. 配置提供商（见配置章节）

4. 重启 ComfyUI

### 基本用法

**图片生成：**
```
添加 flexai:openai_image 节点
→ 设置提供商和模型
→ 输入提示词
→ 生成！
```

**图片编辑：**
```
添加 flexai:openai_image 节点
→ 连接图片到 image_1/2/3/4 输入
→ 设置提供商和模型
→ 输入编辑提示词
→ 编辑！
```

## 配置

在插件根目录创建 `.env` 文件。

### 单一提供商
```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1  # 可选
```

### 多提供商（推荐）
```bash
# 定义提供商列表
OPENAI_PROVIDERS=openai,anthropic,custom

# OpenAI
OPENAI_API_KEY_openai=sk-your-openai-key
OPENAI_API_BASE_openai=https://api.openai.com/v1

# Anthropic (通过 OpenAI 兼容端点)
OPENAI_API_KEY_anthropic=sk-your-anthropic-key  
OPENAI_API_BASE_anthropic=https://api.anthropic.com/v1

# 自定义端点
OPENAI_API_KEY_custom=your-custom-key
OPENAI_API_BASE_custom=https://your-api.example.com/v1
```

### 自动检测
或者，只需定义带后缀的密钥：
```bash
OPENAI_API_KEY_provider1=key1
OPENAI_API_KEY_provider2=key2
```
系统会自动检测并填充提供商下拉菜单。

## 节点参数

### 图片节点 (`flexai:openai_image`)

| 参数 | 类型 | 描述 |
|------|------|------|
| `provider` | 选择 | API 提供商选择 |
| `model` | 字符串 | 模型名称 (如 `dall-e-3`) |
| `prompt` | 字符串 | 生成/编辑提示词 |
| `image_1-4` | 图片 | 可选图片（提供任意张则进入编辑模式） |
| `size` | 字符串 | 输出尺寸 (如 `1024x1024`) |
| `debug` | 布尔 | 启用调试日志 |

### 文本节点 (`flexai:openai_text`)

| 参数 | 类型 | 描述 |
|------|------|------|
| `provider` | 选择 | API 提供商选择 |
| `model` | 字符串 | 模型名称 (如 `gpt-4o`) |
| `system_prompt` | 字符串 | 系统消息 |
| `user_prompt` | 字符串 | 用户消息 |
| `image_1-4` | 图片 | 可选参考图片 |
| `max_tokens` | 整数 | 最大响应令牌数 |
| `temperature` | 浮点 | 采样温度 (0.0-1.0) |
| `stream` | 布尔 | 启用流式模式 |
| `debug` | 布尔 | 启用调试日志 |

## 高级用法

### 图片处理模式

**纯生成：**
- 不连接任何图片
- 使用 `images.generate` 端点
- 适合文本到图片生成

**单图编辑：**
- 连接一张图片到 `image_1`
- 使用 `images.edit` 端点
- 适合风格转换、修改

**多图编辑：**
- 连接 2-4 张图片到 `image_1`, `image_2` 等
- 所有图片作为数组发送到 `images.edit`
- 适合复杂场景编辑

### 错误处理

插件包含强大的错误处理：
- 安全系统拒绝时显示有用指导
- 调试模式下显示网络错误
- 失败操作生成错误可视化图片
- 非阻塞：即使失败工作流也会继续

### 流式文本

启用流式获得实时文本生成：
- 在文本节点中设置 `stream=True`
- 接收增量令牌更新
- 支持时包含使用统计
- 长文本生成的更好用户体验

## 技术细节

### 架构
- **现代 SDK**：基于 OpenAI Python SDK 1.x 构建
- **清晰结构**：使用 `flexai:` 前缀的统一命名空间
- **模块化设计**：独立的图片和文本处理
- **多提供商**：支持任何 OpenAI 兼容端点

### 文件结构
```
Comfyui-flexai/
├── __init__.py                 # 插件注册
├── provider_config.py          # 多提供商管理
├── nodes/
│   ├── flexai/
│   │   ├── openai_image.py    # 图片生成/编辑节点
│   │   └── openai_text.py     # 文本生成节点
│   └── utils/
│       ├── openai_standard.py # OpenAI API 工具
│       └── images.py          # 图片处理工具
├── test/
│   └── test_plugin.py         # 插件测试
├── requirements.txt           # Python 依赖
└── README.md                  # 说明文件
```

### 兼容性

- **ComfyUI**：任何支持自定义节点的最新版本
- **Python**：3.8+ (使用 3.10+ 测试)
- **OpenAI SDK**：1.0+ (优先) 支持 0.x 回退
- **APIs**：任何 OpenAI 兼容端点

## 故障排除

### 常见问题

**插件无法加载：**
- 检查 `requirements.txt` 安装
- 验证 `.env` 文件配置
- 完全重启 ComfyUI

**API 错误：**
- 启用 `debug=True` 获得详细日志
- 验证 API 密钥和端点
- 查看提供商特定文档

**图片处理问题：**
- 确保图片是有效的 ComfyUI 张量
- 检查图片格式兼容性
- 验证模型支持图片处理

**安全拒绝：**
- 查看 OpenAI 使用策略
- 修改提示词内容
- 尝试不同模型变体

### 调试模式

启用调试模式 (`debug=True`) 获得详细日志：
- API 请求/响应详情
- 图片处理步骤
- 错误堆栈跟踪
- 性能指标

## 测试

运行测试套件：
```bash
python -m test.test_plugin
```

这会验证：
- 插件加载
- 提供商配置
- 节点注册
- 基本功能

## 贡献

1. Fork 仓库
2. 创建功能分支
3. 进行更改
4. 添加测试（如适用）
5. 提交 Pull Request

## 许可证

MIT 许可证 - 详见 LICENSE 文件。

## 支持

- **问题报告**：通过 GitHub Issues 报告 bug
- **讨论**：功能请求和一般问题
- **文档**：查看 README.md 英文版本

---

为 ComfyUI 社区用 ❤️ 构建
