@echo off
cd /d C:\Users\chesk\.cursor-tutor\CityPrinter
git add -A
git commit -m "Add print mode toggle to frontend"
git push origin main
del push.bat
