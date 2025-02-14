uint8 ODE     = 0
uint8 BULLET  = 1
uint8 SIMBODY = 2
uint8 DART    = 3

std_msgs/Header header
uint8 type
string solver_type
float64 min_step_size
uint32 precon_iters
uint32 iters
float64 sor
float64 cfm
float64 erp
float64 contact_max_correcting_vel
float64 contact_surface_layer
geometry_msgs/Vector3 gravity
bool enable_physics
float64 real_time_factor
float64 real_time_update_rate
float64 max_step_size
string profile_name
geometry_msgs/Vector3 magnetic_field
