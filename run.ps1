Set-Location .\build
cmake --build . --config release
Set-Location ..
.\build\Release\dcgan.exe