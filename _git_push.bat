chcp 65001 >nul
chdir /d D:\CodeBuddy\MediaPipe\acupoint-platform
git add .
git commit -m "同步项目到GitHub：修复左右手标注镜像问题、更新README位置与内容、新增.gitignore压缩包规则"
git push origin main
del "%~f0"
