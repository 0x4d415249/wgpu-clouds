struct CameraUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
}
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(2.0, 0.0),
        vec2<f32>(0.0, 2.0)
    );
    let uv = uvs[in_vertex_index];
    let pos = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    out.uv = uv;
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    return out;
}

// --- NOISE & MATH ---

fn hash(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Smooth Value Noise (interpolated blocks)
fn noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix( hash(i + vec3<f32>(0.0,0.0,0.0)), 
                        hash(i + vec3<f32>(1.0,0.0,0.0)), u.x),
                   mix( hash(i + vec3<f32>(0.0,1.0,0.0)), 
                        hash(i + vec3<f32>(1.0,1.0,0.0)), u.x), u.y),
               mix(mix( hash(i + vec3<f32>(0.0,0.0,1.0)), 
                        hash(i + vec3<f32>(1.0,0.0,1.0)), u.x),
                   mix( hash(i + vec3<f32>(0.0,1.0,1.0)), 
                        hash(i + vec3<f32>(1.0,1.0,1.0)), u.x), u.y), u.z);
}

// ---------------------------------------------------------
// CLOUD SHAPE
// ---------------------------------------------------------
fn get_cloud_density(p: vec3<f32>) -> f32 {
    // Large scale coordinates
    let scale = 0.015; 
    let pos = p * scale;
    
    // Wind animation
    let wind = vec3<f32>(camera.time * 0.5, 0.0, 0.0);
    let sample_pos = pos + wind;
    
    // Main shape (Blocky feel comes from value noise structure)
    var n = noise(sample_pos);
    
    // Detail (Erosion)
    n += 0.5 * noise(sample_pos * 2.03);
    n /= 1.5;
    
    // Thresholding - The "Minecraft" Cutoff
    // Higher threshold = fewer clouds, sharper edges
    let coverage = 0.5; 
    
    // Smoothstep creates soft volumetric edges. 
    // Tightening these bounds (e.g. 0.5, 0.55) makes them look harder/blockier.
    // Keeping them slightly loose (0.5, 0.65) gives that "Shader Pack" fluffy look.
    let density = smoothstep(coverage, coverage + 0.15, n);
    
    return density;
}

// ---------------------------------------------------------
// RAY-BOX INTERSECTION
// ---------------------------------------------------------
fn intersect_slab(ro: vec3<f32>, rd: vec3<f32>, y_min: f32, y_max: f32) -> vec2<f32> {
    let t0 = (y_min - ro.y) / rd.y;
    let t1 = (y_max - ro.y) / rd.y;
    
    let t_near = min(t0, t1);
    let t_far = max(t0, t1);
    
    return vec2<f32>(t_near, t_far);
}

// ---------------------------------------------------------
// FRAGMENT SHADER
// ---------------------------------------------------------
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // 1. Ray Setup
    let ndc_x = in.uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - in.uv.y * 2.0;
    let clip_pos = vec4<f32>(ndc_x, ndc_y, 1.0, 1.0);
    let world_pos_hom = camera.inv_view_proj * clip_pos;
    let world_pos = world_pos_hom.xyz / world_pos_hom.w;
    let view_dir = normalize(world_pos - camera.camera_pos);
    
    // Sun Setup
    let sun_dir = normalize(vec3<f32>(0.2, 0.5, -0.6));
    
    // --- SKY COLOR (Vibrant Minecraft Style) ---
    // Deep, punchy blue at zenith
    let col_zenith = vec3<f32>(0.05, 0.35, 0.95); 
    // Bright cyan/white at horizon
    let col_horizon = vec3<f32>(0.6, 0.85, 1.0);
    
    // Mix based on angle. We use abs() to keep horizon bright even if looking slightly down
    let horizon_factor = pow(1.0 - max(view_dir.y, 0.0), 4.0);
    var sky_col = mix(col_zenith, col_horizon, horizon_factor);
    
    // Ground color (simple fallback if looking down below clouds)
    if (view_dir.y < -0.01) {
       sky_col = mix(sky_col, vec3<f32>(0.1, 0.1, 0.1), 0.5); // Simple void fog
    }

    var color = sky_col;
    
    // --- VOLUMETRIC CLOUDS ---
    let cloud_bottom = 120.0;
    let cloud_top = 160.0;
    
    // Intersect the cloud layer slab
    let hit = intersect_slab(camera.camera_pos, view_dir, cloud_bottom, cloud_top);
    let t_enter = hit.x;
    let t_exit = hit.y;
    
    // Valid intersection conditions:
    // 1. We must look towards the slab (t_exit > 0)
    // 2. We must overlap with it (t_exit > t_enter)
    if (t_exit > 0.0 && t_exit > t_enter) {
        // Start marching from max(0, t_enter) to handle being INSIDE clouds
        let t_start = max(0.0, t_enter);
        let t_end = t_exit;
        
        // Don't render infinite distance
        if (t_start < 2000.0) {
            let steps = 40;
            // Calculate march length through the volume
            let march_dist = t_end - t_start;
            let step_size = march_dist / f32(steps);
            
            var t = t_start;
            var total_density = 0.0;
            var cloud_color_acc = vec3<f32>(0.0);
            
            // Jitter start position for noise reduction (optional, keeps it simple for now)
            
            for (var i = 0; i < steps; i++) {
                if (total_density >= 1.0) { break; }
                
                let pos = camera.camera_pos + view_dir * t;
                
                // Height Gradient (Soft tops and bottoms)
                // 0.0 at bottom, 1.0 at top
                let h_frac = (pos.y - cloud_bottom) / (cloud_top - cloud_bottom);
                // Parabola fade: 0 -> 1 -> 0
                let h_fade = smoothstep(0.0, 0.2, h_frac) * smoothstep(1.0, 0.8, h_frac);
                
                let dens = get_cloud_density(pos);
                let local_density = dens * h_fade;
                
                if (local_density > 0.001) {
                    // --- LIGHTING MODEL (The "Happy White" Fix) ---
                    
                    // 1. Direct Sun Light
                    // How much sun hits this point?
                    // Simple hack: Top of clouds is brighter
                    let sun_exposure = h_frac; 
                    
                    // 2. Phase Function (Henyey-Greenstein approx)
                    // Makes clouds glow when looking near the sun ("Silver Lining")
                    let cos_angle = dot(view_dir, sun_dir);
                    let phase = 0.5 + 0.5 * pow(0.5 * (cos_angle + 1.0), 6.0);
                    
                    // 3. Powder Effect
                    // Darkens deep internals, Brightens edges
                    let powder = 1.0 - exp(-local_density * 2.0);
                    
                    // Combine Colors
                    // Bright white/yellow sun light
                    let sun_light = vec3<f32>(1.0, 0.98, 0.9) * 2.0 * phase * sun_exposure;
                    // High ambient blue/white light (Fill light) -> Eliminates the "Sad Grey"
                    let ambient = vec3<f32>(0.7, 0.8, 1.0) * 1.2; 
                    
                    // Final scatter color
                    let scatter = mix(ambient, sun_light, 0.5);
                    
                    // Add powder curve to scattering
                    let final_light = scatter * (1.0 + powder * 2.0);
                    
                    let alpha = local_density * 0.4; // Absorption coeff
                    
                    // Front-to-back accumulation
                    let transmittance = 1.0 - total_density;
                    cloud_color_acc += final_light * alpha * transmittance;
                    total_density += alpha;
                }
                
                t += step_size;
            }
            
            // Distance Fog for Clouds
            // Blends clouds into sky color at distance
            let fog_dist = t_start;
            let fog_factor = 1.0 - exp(-fog_dist * 0.0008);
            
            let final_cloud = mix(cloud_color_acc, sky_col, fog_factor);
            
            // Composite Cloud over Sky
            color = mix(color, final_cloud, total_density);
        }
    }
    
    // --- SUN & BLOOM (Post-Cloud) ---
    // Render sun on top if it's not fully obscured (simplified)
    // In a perfect engine, we'd check occlusion, but adding it additively works well for bloom
    let sun_dot = max(dot(view_dir, sun_dir), 0.0);
    // Sharp Core
    let sun_core = smoothstep(0.9992, 0.9999, sun_dot) * 20.0;
    // Wide Glow
    let sun_glow = pow(sun_dot, 20.0) * 0.4;
    
    // Add sun, but masked slightly by cloud density (approximated by checking final color brightness)
    // Actually, just add it. The cloud scattering handles the "look" of sun behind clouds fairly well.
    color += vec3<f32>(1.0, 0.9, 0.7) * (sun_core + sun_glow);

    // --- TONE MAPPING ---
    // Standard ACES-ish curve for punchy contrast
    color = color * 0.9; // Exposure
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    color = clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
    
    return vec4<f32>(color, 1.0);
}
