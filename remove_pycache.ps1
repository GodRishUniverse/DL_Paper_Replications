# Delete all __pycache__ directories
Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
# Delete all .egg-info directories
Get-ChildItem -Path . -Filter "*.egg-info" -Recurse -Directory | Remove-Item -Recurse -Force
