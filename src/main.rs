fn main() {
    let device = metal::Device::system_default().unwrap();
    let lib = device
        .new_library_with_source(include_str!("add.metal"), &metal::CompileOptions::new())
        .unwrap();
    let add_func = lib.get_function("add_arrays", None).unwrap();
    // pipelines define steps the gpu should take to complete a task
    let add_func_pipeline = device
        .new_compute_pipeline_state_with_function(&add_func)
        .unwrap();

    // Works as a scheduler of sorts that enables tasks to be sent via the command buffer
    let command_queue = device.new_command_queue();
    let a: [f32; 3] = [4.7, 9.9, 3.2];
    let b: [f32; 3] = [3.5, 2.6, 1.4];

    let buffer_len = std::mem::size_of::<f32>() * a.len();
    let buffer_a = device.new_buffer_with_data(
        unsafe { std::mem::transmute(a.as_ptr()) },
        buffer_len as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let buffer_b = device.new_buffer_with_data(
        unsafe { std::mem::transmute(b.as_ptr()) },
        buffer_len as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let buffer_c = device.new_buffer(
        buffer_len as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // holds commands to be written to gpu
    let command_buffer = command_queue.new_command_buffer();

    // describes the specific types of commands to be sent
    let command_encoder = command_buffer.new_compute_command_encoder();
    command_encoder.set_compute_pipeline_state(&add_func_pipeline);
    command_encoder.set_buffer(0, Some(&buffer_a), 0);
    command_encoder.set_buffer(1, Some(&buffer_b), 0);
    command_encoder.set_buffer(2, Some(&buffer_c), 0);
    let grid_size = metal::MTLSize::new(buffer_len as u64, 1, 1);
    let thread_group_size =
        if add_func_pipeline.max_total_threads_per_threadgroup() > buffer_len as u64 {
            buffer_len as u64
        } else {
            add_func_pipeline.max_total_threads_per_threadgroup() as u64
        };
    let metal_thread_group_size = metal::MTLSize::new(thread_group_size, 1, 1);
    command_encoder.dispatch_threads(grid_size, metal_thread_group_size);

    command_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    let result = unsafe { &*{ buffer_c.contents() as *mut [f32; 3] } };
    println!("{result:?}");
}
