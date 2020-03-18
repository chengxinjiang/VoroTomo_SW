# Welcome to the *VoroTomo 2D* repository!
This code is specifically designed for 2D surface wave phase/group velocity map inversions, and is built upon the 3D tomography code of Fang et al., (2020). It takes advantage of the Poisson Voronoi projection techinque to remove the *ad hoc* regularization of damping and smoothing that is usually needed in traditional tomography methods. In particular, the code adopts the [pykonal package](https://github.com/malcolmw/pykonal) for traveltime calculation and ray tracying. The pykonal package is a finite-difference solver for the 3D Eikonal equation in both Cartesian and spherical coordinates and is based upon the Fast-Marching Method of Sethian et al. (1996).  
    
    
#### Reference
Fang, H., van der Hilst, R. D., de Hoop, M. V., Kothari, K., Gupta, S., & Dokmanić, I. (2020). "Parsimonious Seismic Tomography with Poisson Voronoi Projections: Methodology and Validation". _Seismological Research Letters_, 91(1), 343-355.

Sethian, J. A. (1996). "A fast marching level set method for monotonically advancing fronts". _*_Proceedings of the National Academy of Sciences_, 93 (4), 1591–1595. 

White, M. C. A., Fang, H., Nakata, N., & Ben-Zion, Y. (2020). "PyKonal: A Python package for solving the Eikonal equation in spherical and Cartesian coordinates using the Fast Marching Method". _Seismological Research Letters_, in review.