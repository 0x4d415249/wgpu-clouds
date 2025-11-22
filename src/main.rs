use wgpu_clouds::run;

fn main() {
    env_logger::init();
    pollster::block_on(run());
}
