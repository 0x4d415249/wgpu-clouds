use crate::world::WorldManager;
use cgmath::Vector3;

#[derive(Clone, Copy, Debug)]
pub struct AABB {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

impl AABB {
    pub fn new(pos: Vector3<f32>, half_extents: Vector3<f32>) -> Self {
        Self {
            min: pos - half_extents,
            max: pos + half_extents,
        }
    }
}

pub struct RaycastResult {
    pub position: [i32; 3],
    pub normal: [i32; 3],
    pub distance: f32,
}

pub fn raycast(
    world: &WorldManager,
    origin: Vector3<f32>,
    dir: Vector3<f32>,
    max_dist: f32,
) -> Option<RaycastResult> {
    let mut x = origin.x.floor() as i32;
    let mut y = origin.y.floor() as i32;
    let mut z = origin.z.floor() as i32;

    let step_x = if dir.x > 0.0 { 1 } else { -1 };
    let step_y = if dir.y > 0.0 { 1 } else { -1 };
    let step_z = if dir.z > 0.0 { 1 } else { -1 };

    let t_delta_x = if dir.x.abs() < 1e-6 {
        f32::INFINITY
    } else {
        (1.0 / dir.x).abs()
    };
    let t_delta_y = if dir.y.abs() < 1e-6 {
        f32::INFINITY
    } else {
        (1.0 / dir.y).abs()
    };
    let t_delta_z = if dir.z.abs() < 1e-6 {
        f32::INFINITY
    } else {
        (1.0 / dir.z).abs()
    };

    let mut t_max_x = if dir.x.abs() < 1e-6 {
        f32::INFINITY
    } else {
        let next_boundary_x = if step_x > 0 { (x + 1) as f32 } else { x as f32 };
        (next_boundary_x - origin.x) / dir.x
    };
    let mut t_max_y = if dir.y.abs() < 1e-6 {
        f32::INFINITY
    } else {
        let next_boundary_y = if step_y > 0 { (y + 1) as f32 } else { y as f32 };
        (next_boundary_y - origin.y) / dir.y
    };
    let mut t_max_z = if dir.z.abs() < 1e-6 {
        f32::INFINITY
    } else {
        let next_boundary_z = if step_z > 0 { (z + 1) as f32 } else { z as f32 };
        (next_boundary_z - origin.z) / dir.z
    };

    let mut normal = [0, 0, 0];
    let mut dist = 0.0;
    let max_steps = (max_dist * 2.0) as i32;

    for _ in 0..max_steps {
        if world.get_block(x, y, z) != 0 {
            return Some(RaycastResult {
                position: [x, y, z],
                normal,
                distance: dist,
            });
        }

        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                x += step_x;
                dist = t_max_x;
                t_max_x += t_delta_x;
                normal = [-step_x, 0, 0];
            } else {
                z += step_z;
                dist = t_max_z;
                t_max_z += t_delta_z;
                normal = [0, 0, -step_z];
            }
        } else if t_max_y < t_max_z {
            y += step_y;
            dist = t_max_y;
            t_max_y += t_delta_y;
            normal = [0, -step_y, 0];
        } else {
            z += step_z;
            dist = t_max_z;
            t_max_z += t_delta_z;
            normal = [0, 0, -step_z];
        }
        if dist > max_dist {
            break;
        }
    }
    None
}

pub fn resolve_collision(
    world: &WorldManager,
    pos: Vector3<f32>,
    velocity: Vector3<f32>,
    player_dims: Vector3<f32>,
) -> (Vector3<f32>, Vector3<f32>, bool) {
    let mut new_pos = pos;
    let mut new_vel = velocity;
    let mut on_ground = false;

    let half_size = player_dims / 2.0;
    let epsilon = 0.001;

    // Y Axis
    if new_vel.y != 0.0 {
        new_pos.y += new_vel.y;
        let aabb = AABB::new(new_pos, half_size);
        if check_collision(world, &aabb) {
            if new_vel.y < 0.0 {
                new_pos.y = aabb.min.y.floor() + 1.0 + half_size.y + epsilon;
                on_ground = true;
            } else {
                new_pos.y = aabb.max.y.floor() - half_size.y - epsilon;
            }
            new_vel.y = 0.0;
        }
    }

    // X Axis
    if new_vel.x != 0.0 {
        new_pos.x += new_vel.x;
        let aabb = AABB::new(new_pos, half_size);
        if check_collision(world, &aabb) {
            if new_vel.x > 0.0 {
                new_pos.x = aabb.max.x.floor() - half_size.x - epsilon;
            } else {
                new_pos.x = aabb.min.x.floor() + 1.0 + half_size.x + epsilon;
            }
            new_vel.x = 0.0;
        }
    }

    // Z Axis
    if new_vel.z != 0.0 {
        new_pos.z += new_vel.z;
        let aabb = AABB::new(new_pos, half_size);
        if check_collision(world, &aabb) {
            if new_vel.z > 0.0 {
                new_pos.z = aabb.max.z.floor() - half_size.z - epsilon;
            } else {
                new_pos.z = aabb.min.z.floor() + 1.0 + half_size.z + epsilon;
            }
            new_vel.z = 0.0;
        }
    }

    (new_pos, new_vel, on_ground)
}

fn check_collision(world: &WorldManager, aabb: &AABB) -> bool {
    let min_x = aabb.min.x.floor() as i32;
    let max_x = aabb.max.x.floor() as i32;
    let min_y = aabb.min.y.floor() as i32;
    let max_y = aabb.max.y.floor() as i32;
    let min_z = aabb.min.z.floor() as i32;
    let max_z = aabb.max.z.floor() as i32;

    for y in min_y..=max_y {
        for z in min_z..=max_z {
            for x in min_x..=max_x {
                if world.get_block(x, y, z) != 0 {
                    return true;
                }
            }
        }
    }
    false
}
