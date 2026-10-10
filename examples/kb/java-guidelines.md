# Java 代码规范

## 空值处理

- 对外接口返回值建议统一用 `Result<T>` 包装，而不是直接返回可能为 null 的对象。
- 判断字符串相等用 `"constant".equals(obj)` 或 `Objects.equals(a, b)`，避免空指针。
- 方法入参如果是必填，应在入口处校验，而不是在深处才出现 NPE。

## 异常与事务

- Service 层不要吞异常，事务方法里异常要正确回滚；禁止 `catch (Exception e) {}` 空处理。
- 不要用异常做业务分支控制；区分业务异常和系统异常，统一在全局异常处理器转成规范响应。
- 事务边界尽量在 Service 层，避免在 Controller 或循环里开多个事务。

## 资源与线程

- 数据库连接、流、文件必须在 finally 或 try-with-resources 中关闭。
- 线程池不要用 `Executors.newCachedThreadPool`（无界队列），用 `ThreadPoolExecutor` 并显式指定参数。
- 线程池任务要设置超时，防止任务卡死耗尽线程。

## 集合与并发

- 遍历集合时删除元素用迭代器的 `remove()`，避免 `ConcurrentModificationException`。
- 共享可变状态用 `ConcurrentHashMap`、`AtomicInteger` 等并发容器或加锁保护。
- 能用不可变集合（如 `Collections.unmodifiableList`）的地方优先不可变，减少意外修改。

## 日志

- 日志不要用字符串拼接：用占位符 `log.info("order={}, user={}", id, uid)`。
- 异常日志要带堆栈：`log.error("xxx", e)`，不要把 `e.getMessage()` 单独丢掉。
- 不要在高频路径打印大对象或敏感信息（密码、token）。

## 命名与分层

- Controller 只做参数校验和转发，业务逻辑放 Service，数据访问放 Mapper/Repository。
- 类名用名词、方法名用动词，布尔方法用 `isXxx` / `hasXxx`。
