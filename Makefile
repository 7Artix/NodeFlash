# 定义目录和文件模式
METAL_DIR = metal
METAL_SOURCES = $(wildcard $(METAL_DIR)/*.metal)      # 查找所有 .metal 文件
METAL_AIRS = $(METAL_SOURCES:.metal=.air)            # 对应的 .air 文件
METAL_LIBS = $(METAL_AIRS:.air=.metallib)            # 对应的 .metallib 文件
SWIFT_OUTPUT_DIR = bin

# Swift 可执行文件名（默认为 Package.swift 中定义的名称）
SWIFT_EXEC = NodeFlash

# 默认目标：编译所有 Metal 文件和 Swift 工程
all: build

# 生成 .air 文件的规则（自动匹配多个 .metal 文件）
$(METAL_DIR)/%.air: $(METAL_DIR)/%.metal | $(METAL_DIR)
	xcrun -sdk macosx metal -c $< -o $@

# 生成 .metallib 文件的规则（自动匹配多个 .air 文件）
$(METAL_DIR)/%.metallib: $(METAL_DIR)/%.air
	xcrun -sdk macosx metallib $< -o $@

# 编译 Swift 工程，依赖于所有 .metallib 文件
build: $(METAL_LIBS)
	swift build --build-path $(SWIFT_OUTPUT_DIR)
	mv $(SWIFT_OUTPUT_DIR)/debug/$(SWIFT_EXEC) $(SWIFT_OUTPUT_DIR)/$(SWIFT_EXEC)

# 运行 Swift 可执行文件
run: build
	$(SWIFT_OUTPUT_DIR)/$(SWIFT_EXEC)

# 清理生成的文件
clean:
	rm -f $(METAL_DIR)/*.air $(METAL_DIR)/*.metallib
	swift package clean