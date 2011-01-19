@echo off
@for %%f in (%1) do call :make_one %%f
@exit /B 0


:make_one
@echo off
@if '%1' == 'main.cpp' exit /B 0

@echo Generating %~n1.cu

@echo #include ".\%~n1.cpp" > %~n1.cu
@echo #ifdef __CUDACC__ >> %~n1.cu
@echo #include "main.cpp" >> %~n1.cu
@echo #endif >> %~n1.cu
@echo //////////////////////////////////////// >> %~n1.cu

@echo ...Done
@echo -

@exit /B 0