#!/bin/bash

# AI Toolkit 部署脚本
echo "开始部署 AI Toolkit..."

# 拉取最新代码
echo "正在拉取最新代码..."
git pull origin main

# 检查是否需要安装依赖
if [ -f "ui/package.json" ]; then
    echo "检查依赖..."
    cd ui
    
    # 检查package-lock.json是否有变化
    if [ -n "$(git diff HEAD~1 HEAD -- package-lock.json)" ]; then
        echo "检测到依赖变化，正在安装..."
        npm install
    fi
    
    # 构建项目
    echo "正在构建项目..."
    npm run build
    
    # 停止旧服务（如果使用PM2）
    if command -v pm2 &> /dev/null; then
        echo "停止旧的服务..."
        pm2 stop ai-toolkit-ui 2>/dev/null || true
    fi
    
    # 启动服务
    echo "启动服务..."
    if command -v pm2 &> /dev/null; then
        # 使用PM2启动（推荐）
        pm2 start npm --name "ai-toolkit-ui" -- start
        pm2 save
    else
        # 直接启动
        npm start
    fi
    
    cd ..
fi

echo "部署完成！"
echo "项目应该在 http://your-server-ip:3000 访问"