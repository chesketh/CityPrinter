@echo off
cd /d C:\Users\chesk\.cursor-tutor\CityPrinter
git add -A
git commit --amend -m "CityPrinter initial commit"
git push -u origin main --force
del push2.bat
